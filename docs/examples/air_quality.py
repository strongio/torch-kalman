# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"nbsphinx": "hidden"}
import torch
import copy

from torchcast.utils.datasets import load_air_quality_data
from torchcast.kalman_filter import KalmanFilter
from torchcast.utils.data import TimeSeriesDataset

import numpy as np
import pandas as pd

np.random.seed(2021-1-21)
torch.manual_seed(2021-1-21)
# -

# # Multivariate Forecasts: Beijing Multi-Site Air-Quality Data
#
# We'll demonstrate several features of `torchcast` using a dataset from the [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data). It includes data on air pollutants and weather from 12 sites.

# + {"tags": ["remove_cell"]}
df_aq = load_air_quality_data('weekly')

SPLIT_DT = np.datetime64('2016-02-22')

df_aq

# + [markdown] {"hidePrompt": true}
# ### Univariate Forecasts
#
# Let's try to build a model to predict total particulate-matter (PM2.5 and PM10). 
#
# First, we'll make our target the sum of these two types. We'll log-transform since this is strictly positive.

# +
from torchcast.process import LocalTrend, Season

# create a dataset:
df_aq['PM'] = df_aq['PM10'] + df_aq['PM2p5'] 
df_aq['PMlog10'] = np.log10(df_aq['PM']) # TODO: underscore
dataset_pm_univariate = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq,
    dt_unit='W',
    measure_colnames=['PMlog10'],
    group_colname='station', 
    time_colname='week'
)
dataset_pm_univariate_train, _ = dataset_pm_univariate.train_val_split(dt=SPLIT_DT)

# create a model:
kf_pm_univariate = KalmanFilter(
    measures=['PMlog10'], 
    processes=[
        LocalTrend(id='trend'),
        Season(id='day_in_year', period=365.25 / 7, dt_unit='W', K=5)
    ]
)

# fit:
kf_pm_univariate.fit(
    dataset_pm_univariate_train.tensors[0],
    start_datetimes=dataset_pm_univariate_train.start_datetimes
)


# -

# Let's see how our forecasts look:

# +
# helper for transforming log back to original:
def inverse_transform(df):
    df = df.copy(deep=False)
    # bias-correction for log-transform (see https://otexts.com/fpp2/transformations.html#bias-adjustments)
    df['mean'] += .5 * (df['upper'] - df['lower']) / 1.96 ** 2
    # inverse the log10:
    df[['actual', 'mean', 'upper', 'lower']] = 10 ** df[['actual', 'mean', 'upper', 'lower']]
    df['measure'] = df['measure'].str.replace('_log10', '')
    return df

# generate forecasts:
forecast = kf_pm_univariate(
        dataset_pm_univariate_train.tensors[0],
        start_datetimes=dataset_pm_univariate_train.start_datetimes,
        out_timesteps=dataset_pm_univariate.tensors[0].shape[1]
)

df_forecast = inverse_transform(forecast.to_dataframe(dataset_pm_univariate))
print(forecast.plot(df_forecast, max_num_groups=3, split_dt=SPLIT_DT))
# -

# #### Evaluating Performance: Expanding Window
#
#
# TODO: explain

# +
pred_4step = kf_pm_univariate(
    dataset_pm_univariate.tensors[0],
    start_datetimes=dataset_pm_univariate.start_datetimes,
    n_step=4
)

df_univariate_error = pred_4step.\
    to_dataframe(dataset_pm_univariate, group_colname='station', time_colname='week').\
    pipe(inverse_transform).\
    merge(df_aq.loc[:,['station', 'week', 'PM']]).\
    assign(
        error = lambda df: np.abs(df['mean'] - df['actual']),
        validation = lambda df: df['week'] > SPLIT_DT
    ).\
    groupby(['station','validation'])\
    ['error'].mean().\
    reset_index()
df_univariate_error.groupby('validation')['error'].agg(['mean','std'])
# -

# ### Multivariate Forecasts
#
# Can we improve our moodel by splitting the pollutant we are forecasting into its two types (2.5 and 10), and modeling them in a multivariate manner?

# +
# create a dataset:
df_aq['PM10_log10'] = np.log10(df_aq['PM10'])
df_aq['PM2p5_log10'] = np.log10(df_aq['PM2p5'])
dataset_pm_multivariate = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq,
    dt_unit='W',
    measure_colnames=['PM10_log10','PM2p5_log10'],
    group_colname='station', 
    time_colname='week'
)
dataset_pm_multivariate_train, _ = dataset_pm_multivariate.train_val_split(dt=SPLIT_DT)

# create a model:
_processes = []
for m in dataset_pm_multivariate.measures[0]:
    _processes.extend([
        LocalTrend(id=f'{m}_trend', measure=m),
        Season(id=f'{m}_day_in_year', period=365.25 / 7, dt_unit='W', K=5, measure=m)
    ])
kf_pm_multivariate = KalmanFilter(measures=dataset_pm_multivariate.measures[0], processes=_processes)

# fit:
kf_pm_multivariate.fit(
    dataset_pm_multivariate_train.tensors[0],
    start_datetimes=dataset_pm_multivariate_train.start_datetimes
)

# +
with torch.no_grad():
    pred_4step = kf_pm_multivariate(
        dataset_pm_multivariate.tensors[0],
        start_datetimes=dataset_pm_multivariate.start_datetimes,
        n_step=4
    )
    # generate draws from the forecast distribution:
    mc_draws = 10 ** torch.distributions.MultivariateNormal(*pred_4step).rsample((500,))
    # sum across 2.5 and 10, then mean across draws:
    mc_predictions = mc_draws.sum(-1, keepdim=True).mean(0)
    
# convert to a dataframe and summarize error:
_df_pred = TimeSeriesDataset.tensor_to_dataframe(
    mc_predictions, 
    times=dataset_pm_multivariate.times(),
    group_names=dataset_pm_multivariate.group_names,
    group_colname='station',
    time_colname='week',
    measures=['predicted']
)    
df_multivariate_error = _df_pred.\
    merge(df_aq.loc[:,['station', 'week', 'PM']]).\
    assign(
        error = lambda df: np.abs(df['predicted'] - df['PM']),
        validation = lambda df: df['week'] > SPLIT_DT
    ).\
    groupby(['station','validation'])\
    ['error'].mean().\
    reset_index()
df_multivariate_error.groupby('validation')['error'].agg(['mean','std'])
# -

df_multivariate_error.\
    merge(df_univariate_error, on=['station', 'validation']).\
    assign(error_diff = lambda df: df['error_x'] - df['error_y']).\
    boxplot('error_diff', by='validation')

# ### Incorporating Predictors 

# +
from torchcast.process import LinearModel

# create lagged predictors:
predictors = pd.Series(['TEMP','PRES', 'DEWP'])
df_lagged_preds = df_aq.set_index('week').groupby('station')[predictors].shift(4, freq='7D')
df_lagged_preds.columns = df_lagged_preds.columns + '_lag4'
predictors = df_lagged_preds.columns
df_lagged_preds -= df_lagged_preds.mean()
df_lagged_preds /= df_lagged_preds.std()

# create a dataset:
dataset_pm_multivariate = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq.merge(df_lagged_preds.reset_index()),
    dt_unit='W',
    y_colnames=['PM10_log10','PM2p5_log10'],
    X_colnames=predictors,
    group_colname='station', 
    time_colname='week'
)
dataset_pm_multivariate_train, _ = dataset_pm_multivariate.train_val_split(dt=SPLIT_DT)

# create a model:
_processes = []
for m in dataset_pm_multivariate.measures[0]:
    _processes.extend([
        LocalTrend(id=f'{m}_trend', measure=m),
        LinearModel(id=f'{m}_lm', measure=m, predictors=predictors),
        Season(id=f'{m}_day_in_year', period=365.25 / 7, dt_unit='W', K=5, measure=m)
    ])
kf_pm_lm = KalmanFilter(measures=dataset_pm_multivariate.measures[0], processes=_processes)

# fit:
y, X = dataset_pm_multivariate_train.tensors
kf_pm_lm.fit(
    y,
    X=X,
    start_datetimes=dataset_pm_multivariate_train.start_datetimes
)

# +
with torch.no_grad():
    y, X = dataset_pm_multivariate.tensors
    pred_4step = kf_pm_lm(
        y,
        X=X,
        start_datetimes=dataset_pm_multivariate.start_datetimes,
        n_step=4
    )
    mc_draws = 10 ** torch.distributions.MultivariateNormal(*pred_4step).rsample((500,))
    mc_predictions = mc_draws.sum(-1, keepdim=True).mean(0)
_df_pred = TimeSeriesDataset.tensor_to_dataframe(
    mc_predictions, 
    times=dataset_pm_multivariate.times(),
    group_names=dataset_pm_multivariate.group_names,
    group_colname='station',
    time_colname='week',
    measures=['predicted']
)    

df_mv_lm_error = _df_pred.\
    merge(df_aq.loc[:,['station', 'week', 'PM']]).\
    assign(
        error = lambda df: np.abs(df['predicted'] - df['PM']),
        validation = lambda df: df['week'] > SPLIT_DT
    ).\
    groupby(['station','validation'])\
    ['error'].mean().\
    reset_index()
df_mv_lm_error.groupby('validation')['error'].agg(['mean','std'])
# -

df_mv_lm_error.\
    merge(df_multivariate_error, on=['station', 'validation']).\
    assign(error_diff = lambda df: df['error_x'] - df['error_y']).\
    boxplot('error_diff', by='validation')

# ### Predicting State with a Neural Network

# +
from torchcast.process import NN

# XXX
state_nn = torch.nn.Sequential(torch.nn.Linear(len(predictors), 10), torch.nn.Tanh(), torch.nn.Linear(10, 2))

_processes = []
for m in dataset_pm_multivariate.measures[0]:
    _processes.extend([
        LocalTrend(id=f'{m}_trend', measure=m),
        NN(id=f'{m}_nn', measure=m, nn=state_nn),
        Season(id=f'{m}_day_in_year', period=365.25 / 7, dt_unit='W', K=5, measure=m)
    ])
kf_pm_nn = KalmanFilter(measures=dataset_pm_multivariate.measures[0], processes=_processes)


# -

class EarlyStopping:
    def __init__(self, 
                 kf: KalmanFilter, 
                 holdout_dataset: TimeSeriesDataset, 
                 recent_len: int = 3, 
                 tol: float = .001):
        self.kf = kf
        self.holdout_dataset = holdout_dataset
        self.lh = []
        assert recent_len > 1
        self.recent_len = recent_len
        self.tol = tol
    
    @torch.no_grad()
    def __call__(self, *args):
        y, X = self.holdout_dataset.tensors
        pred = self.kf(y, X=X, start_datetimes=self.holdout_dataset.start_datetimes)
        self.lh.append(-pred.log_prob(y).mean().item())
        if len(self.lh) >= self.recent_len:
            loss_decrease = -pd.Series(self.lh[-self.recent_len:]).diff()
            if loss_decrease.max() < self.tol:
                print(f"Holdout loss has not improved in last {self.recent_len-1}, stopping")
                raise KeyboardInterrupt


# +

# fit:
y, X = dataset_pm_multivariate_train[0:11].tensors
kf_pm_nn.fit(
    y,
    X=X,
    start_datetimes=dataset_pm_multivariate_train.start_datetimes[0:11],
    callbacks=[EarlyStopping(kf=kf_pm_nn, holdout_dataset=dataset_pm_multivariate_train[11:])]
)

# +
with torch.no_grad():
    y, X = dataset_pm_multivariate.tensors
    pred_4step = kf_pm_nn(
        y,
        X=X,
        n_step=4,
        start_datetimes=dataset_pm_multivariate.start_datetimes
    )
    mc_draws = 10 ** torch.distributions.MultivariateNormal(*pred_4step).rsample((500,))
    mc_predictions = mc_draws.sum(-1, keepdim=True).mean(0)
_df_pred = TimeSeriesDataset.tensor_to_dataframe(
    mc_predictions, 
    times=dataset_pm_multivariate.times(),
    group_names=dataset_pm_multivariate.group_names,
    group_colname='station',
    time_colname='week',
    measures=['predicted']
)    

df_mv_nn_error = _df_pred.\
    merge(df_aq.loc[:,['station', 'week', 'PM']]).\
    assign(
        error = lambda df: np.abs(df['predicted'] - df['PM']),
        validation = lambda df: df['week'] > SPLIT_DT
    ).\
    groupby(['station','validation'])\
    ['error'].mean().\
    reset_index()
df_mv_nn_error.groupby('validation')['error'].agg(['mean','std'])
# -

df_mv_nn_error.\
    merge(df_multivariate_error, on=['station', 'validation']).\
    assign(error_diff = lambda df: df['error_x'] - df['error_y']).\
    boxplot('error_diff', by='validation')

)

_processes = []
    _processes.extend([
        LocalTrend(id=f'{m}_trend', measure=m),
    ])
kf_pm_nn2 = KalmanFilter(
    measures=dataset_pm_multivariate.measures[0], 
    processes=_processes,
    measure_covariance=Covariance.for_measures(
        measures=dataset_pm_multivariate.measures[0], predict_variance=predict_mvar
    ),
    process_covariance=Covariance.for_processes(
        processes=_processes, predict_variance=predict_pvar
    )
)

# fit:
    y, 
    X=X, 
)

# +
with torch.no_grad():
        y,
        X=X,
    )
    mc_draws = 10 ** torch.distributions.MultivariateNormal(*pred_4step).rsample((500,))
    mc_predictions = mc_draws.sum(-1, keepdim=True).mean(0)
_df_pred = TimeSeriesDataset.tensor_to_dataframe(
    mc_predictions, 
    group_colname='station',
    time_colname='week',
    measures=['predicted']
)    

    merge(df_aq.loc[:,['station', 'week', 'PM']]).\
# -



