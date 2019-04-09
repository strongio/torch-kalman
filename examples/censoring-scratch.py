# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"hideCode": false, "hidePrompt": false}
# %matplotlib inline

from IPython.display import clear_output
from matplotlib import pyplot as plt

import torch

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, LocalTrend
from torch_kalman.state_belief.families import CensoredGaussian
from torch_kalman.state_belief.over_time import StateBeliefOverTime
import pandas as pd
import numpy as np
from plotnine import *

from typing import Optional, Tuple, Callable, Union
# -

"""
TODO:
- sim process is also a local-trend, but then manually add increase as a fxn of log(time)
- make a KF that combines its predictions with that of an NN
"""


# +
class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.design_params = torch.nn.ParameterDict([('test1', torch.nn.Parameter(torch.randn(1))),
                                                     ('test2', torch.nn.Parameter(torch.randn(1)))])                                             
list(Test().named_parameters())


# -



# + {"hideCode": false, "hidePrompt": false}
def get_sim_kf():
    processes = [LocalTrend(id='local_level', decay_velocity=False, multi=0.01)]
    processes[-1].add_measure('y')
    sim_kf = KalmanFilter(measures=['y'], processes=processes)
    sim_kf.design.process_cholesky_log_diag.data[0] -= 4.
    sim_kf.design.process_cholesky_log_diag.data[1] -= 9.
    sim_kf.design.init_state_mean_params.data[0] = 0.
    sim_kf.design.init_state_mean_params.data[1] = 4.
    sim_kf.design.init_cholesky_log_diag.data[0] -= 6.
    sim_kf.design.init_cholesky_log_diag.data[0] -= .5
    return sim_kf

def simulate_data(sim_kf: Optional[KalmanFilter] = None, 
                  num_iter: int = 1, 
                  num_timesteps: int = 200) -> torch.Tensor:
    if sim_kf is None:
        sim_kf = get_sim_kf()
    
    with torch.no_grad():
        design_for_batch = sim_kf.design_for_batch(num_groups=1, num_timesteps=num_timesteps)
        sim = sim_kf.simulate(sim_kf.predict_initial_state(design_for_batch),
                              horizon=design_for_batch.num_timesteps,
                              num_iter=num_iter,
                              progress=True)
        sim = torch.cat(sim, 0)
    return sim

def censor_simulated_data(sim: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    sim_cens = sim.clone()
    
    sim_cens[(sim > 2.)] = 2.
    upper = 2. * torch.ones_like(sim_cens)
    
    sim_cens[(sim < -2.)] = -2.
    lower = -2. * torch.ones_like(sim_cens)
    
    return sim_cens, lower, upper

def tensor_to_df(sim: torch.Tensor) -> pd.DataFrame:
    if sim.shape[2] > 1:
        raise NotImplementedError()
    dfs = []
    for i in range(sim.shape[0]):
        df = pd.DataFrame({'y' : sim[i].squeeze()})
        df['time'] = df.index
        df['group'] = str(i)
        dfs.append(df)
        
    return pd.concat(dfs)

def quick_kf(lr: float, family=None) -> KalmanFilter:
    if family:
        Cls = type(family.__name__ + 'KF', (KalmanFilter, ), {'family' : family})
    else:
        Cls = KalmanFilter
    #processes = [LocalTrend(id='local_level', decay_velocity=False, multi=0.01)]
    processes = [LocalLevel(id='local_level')]
    processes[-1].add_measure('y')
    kf = Cls(measures=['y'], processes=processes)
    kf.design.process_cholesky_log_diag.data[:] -= 1.
    kf.design.init_state_mean_params.data[:] = 0.
    kf.design.init_cholesky_log_diag.data[:] -= 6.
    kf.optimizer = torch.optim.Adam(kf.parameters(), lr=lr)
    return kf

def train_iter(kf: KalmanFilter, 
               input: Union[torch.Tensor, Tuple], 
               step: bool = True, 
               get_loss: Optional[Callable] = None) -> StateBeliefOverTime:    
    
    pred = kf(input)
    if isinstance(input, torch.Tensor):
        log_prob = pred.log_prob(input)
    else:
        log_prob = pred.log_prob(*input)
    loss = -log_prob.mean()

    kf.optimizer.zero_grad()
    loss.backward()
    if step:
        print([param.grad.data for param in kf.parameters()])
        #torch.nn.utils.clip_grad_value_(kf.parameters(), 1.0)
        kf.optimizer.step()
    return pred


# + {"hideCode": false, "hidePrompt": false}
np.random.seed(2019-3-10)
torch.manual_seed(2019-3-10)
sim1 = simulate_data(num_iter=5000, num_timesteps=20)
print(
    ggplot(tensor_to_df(sim1).query("group.isin(group.sample(3))"),
       aes(x='time', y='y', group='group', color='group')) + geom_line()
)

# + {"hideCode": false, "hidePrompt": false}
sim1_cens, sim1_lower, sim1_upper = censor_simulated_data(sim1)
print(
    ggplot(tensor_to_df(sim1_cens).query("group.isin(group.sample(3))"),
       aes(x='time', y='y', group='group', color='group')) + geom_line()
)

# + {"hideCode": false, "hidePrompt": false}
is_cens = (sim1_cens==sim1_lower) | (sim1_cens==sim1_upper)
sim1_cens_args = (sim1_cens, sim1_lower, sim1_upper)

# + {"hideCode": false, "hidePrompt": false}
kf_cens = quick_kf(lr=.05, family=CensoredGaussian)
kf_cens.loss_history1 = []
kf_cens.loss_history2 = []
kf_cens.true_mse_history1 = []
kf_cens.true_mse_history2 = []

for i in range(150):
    pred = train_iter(kf_cens, input=sim1_cens_args)
    kf_cens_sq_err = (pred.predictions - sim1) ** 2
    
    if i > 1:
        loss1 = -pred.log_prob(*sim1_cens_args, method='update').mean().item()
        kf_cens.loss_history1.append(loss1)
        kf_cens.loss_history2.append(-pred.log_prob(*sim1_cens_args, method='independent').mean().item())

        kf_cens.true_mse_history1.append(kf_cens_sq_err[is_cens.squeeze(-1)].mean().item())
        kf_cens.true_mse_history2.append(kf_cens_sq_err[~is_cens.squeeze(-1)].mean().item())

        clear_output(wait=True)
        #plt.plot(kf_cens.loss_history1)
        plt.plot(kf_cens.loss_history2)
        #plt.legend(labels=['update', 'independent'])
        plt.title('loss')
        plt.show()

        plt.plot(kf_cens.true_mse_history1)
        plt.plot(kf_cens.true_mse_history2)
        plt.legend(labels=['censored', 'uncensored'])
        plt.title('mse (ground truth)')
        plt.show()

# + {"hideCode": false, "hidePrompt": false}
df = pd.concat([tensor_to_df(sim1_cens).assign(type='observed'),
                tensor_to_df(sim1).assign(type='actual'),
                tensor_to_df(pred.predictions.detach()).assign(type='predicted'),
                #tensor_to_df(torch.pow(pred.predictions - sim1, 2).detach()).assign(type='mse')
               ])
df = df.merge(tensor_to_df(pred.prediction_uncertainty.detach()).assign(type='predicted', std=lambda df: np.sqrt(df.pop('y'))),
             how='left')
# -

print(
    ggplot(df.query('group.isin(group.sample())'), aes(x='time', y='y', group='type', color='type')) +
    geom_line() +
    geom_ribbon(aes(ymin='y-std', ymax='y+std'), alpha=.25)
)


