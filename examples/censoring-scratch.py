# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
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

from typing import Optional, Tuple


# + {"hideCode": false, "hidePrompt": false}
def simulate_data(sim_kf: Optional[KalmanFilter] = None, 
                  num_iter: int = 1, 
                  num_timesteps: int = 300) -> torch.Tensor:
    if sim_kf is None:
        processes = [LocalTrend(id='local_level', decay_velocity=False, multi=0.01)]
        processes[-1].add_measure('y')
        sim_kf = KalmanFilter(measures=['y'], processes=processes)
        sim_kf.design.process_cholesky_log_diag.data[0] -= 4.
        sim_kf.design.process_cholesky_log_diag.data[1] -= 9.
        sim_kf.design.init_state_mean_params.data[0] = -3.
        sim_kf.design.init_state_mean_params.data[1] = 2.
        sim_kf.design.init_cholesky_log_diag.data -= 6.
    
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
    sim_cens[(sim > 0.)] = 0.
    upper = torch.zeros_like(sim_cens)
    lower = -np.inf * torch.ones_like(sim_cens)
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
    processes = [LocalTrend(id='local_level', multi=0.01)]
    processes[-1].add_measure('y')
    kf = Cls(measures=['y'], processes=processes)
    kf.design.process_cholesky_log_diag.data[:] -= 1.
    kf.design.init_state_mean_params.data[:] = 0.
    kf.design.init_cholesky_log_diag.data[:] -= 6.
    kf.optimizer = torch.optim.Adam(kf.parameters(), lr=lr)
    return kf

def train_iter(kf: KalmanFilter, input: torch.Tensor, step: bool = True) -> Tuple:
    pred = kf(input)
    if isinstance(input, torch.Tensor):
        log_prob = pred.log_prob(input)
    else:
        log_prob = pred.log_prob(*input)
    loss = -log_prob.mean()

    kf.optimizer.zero_grad()
    loss.backward()
    if step:
        kf.optimizer.step()
    return loss.item(), pred


# + {"hideCode": false, "hidePrompt": false}
np.random.seed(2019-3-10)
torch.manual_seed(2019-3-10)
sim1 = simulate_data(num_iter=50)
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
# # no censoring:
# kf_nocens = quick_kf(lr=.01)
# kf_nocens.loss_history = []
# kf_nocens.true_mse_history = []

# sim1_delete = sim1_cens.clone()
# sim1_delete[(sim1_delete==sim1_lower) | (sim1_delete==sim1_upper)] = np.nan

# # censoring:
# kf_cens = quick_kf(lr=.01, family=CensoredGaussian)
# kf_cens.loss_history = []
# kf_cens.true_mse_history = []
# sim1_cens_args = (sim1_cens, sim1_lower, sim1_upper)

for i in range(166, 250):
    # just delete data:
    kf_nocens_loss, kf_nocens_pred = train_iter(kf_nocens, sim1_delete)
    kf_nocens.loss_history.append(kf_nocens_loss)
    kf_nocens_sq_err = (kf_nocens_pred.predictions - sim1) ** 2
    kf_nocens.true_mse_history.append(kf_nocens_sq_err.mean().log().item())
    
    # use censoring:
    kf_cens_loss, kf_cens_pred = train_iter(kf_cens, sim1_cens_args)
    kf_cens.loss_history.append(kf_cens_loss)
    kf_cens_sq_err = (kf_cens_pred.predictions - sim1) ** 2
    kf_cens.true_mse_history.append(kf_cens_sq_err.mean().log().item())
    
    clear_output(wait=True)
    if i > 5:
        plt.plot(kf_nocens.loss_history[5:])
        plt.plot(kf_cens.loss_history[5:])
        plt.legend(labels=['kf_nocens','kf_cens'])
        plt.title('loss')
        plt.show()
        plt.plot(kf_nocens.true_mse_history[5:])
        plt.plot(kf_cens.true_mse_history[5:])
        plt.legend(labels=['kf_nocens','kf_cens'])
        plt.title('mse (ground truth)')
        plt.show()
        
        
        if (i % 5) == 0:
            foo = pd.concat([tensor_to_df(sim1).assign(type='actual'),
                             tensor_to_df(sim1_cens).assign(type='observed'),
                             tensor_to_df(kf_cens(sim1_cens_args).predictions.detach()).assign(type='pred_cens'),
                             tensor_to_df(kf_nocens(sim1_delete).predictions.detach()).assign(type='pred_delete')])
            bar = pd.concat([
                tensor_to_df(kf_cens(sim1_cens_args).prediction_uncertainty.squeeze(-1).detach()).assign(type='pred_cens'),
                tensor_to_df(kf_nocens(sim1_delete).prediction_uncertainty.squeeze(-1).detach()).assign(type='pred_delete')
            ]).assign(std = lambda df: np.sqrt(df.pop('y')))
            foo = foo.merge(bar, how='left')

            print(
                ggplot(foo.query("group == '3'"),
                         aes(x='time', y='y', group='type', color='type')) + 
                  geom_line(size=1.5) +
                  geom_ribbon(aes(ymin='y - std', ymax='y + std'), 
                              alpha=.20,
                              data=lambda df: df.query("type=='pred_cens'"))
                 )

# + {"hideCode": false, "hidePrompt": false}


# + {"hideCode": false, "hidePrompt": false}

