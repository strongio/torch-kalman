# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
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
df_aq['PM_log10'] = np.log10(df_aq['PM']) 
dataset_pm_univariate = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq,
    dt_unit='W',
    measure_colnames=['PM_log10'],
    group_colname='station', 
    time_colname='week'
)
dataset_pm_univariate_train, _ = dataset_pm_univariate.train_val_split(dt=SPLIT_DT)

# create a model:
kf_pm_univariate = KalmanFilter(
    measures=['PM_log10'], 
    processes=[
        LocalTrend(id='trend'),
        Season(id='day_in_year', period=365.25 / 7, dt_unit='W', K=5)
    ]
)

# fit:
kf_pm_univariate.fit(
    dataset_pm_univariate_train.tensors[0],
    start_offsets=dataset_pm_univariate_train.start_datetimes
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
        start_offsets=dataset_pm_univariate_train.start_datetimes,
        out_timesteps=dataset_pm_univariate.tensors[0].shape[1]
)

df_forecast = inverse_transform(forecast.to_dataframe(dataset_pm_univariate))
print(forecast.plot(df_forecast, max_num_groups=3, split_dt=SPLIT_DT))
# -

# #### Evaluating Performance: Expanding Window
#
#
# To evaluate our forecasts, we will not use the long-range forecasts above. Instead, we will use an [expanding window](https://eng.uber.com/forecasting-introduction#:~:text=Comparing) approach to evaluate a shorter forecast horizon. In this approach, we generate N-step-ahead forecasts at every timepoint:
#
# ![title](expanding_window.png)
#
#
# This approach is straightforward in `torchcast`, using the `n_step` argument. Here we'll generate 4-week-ahead predictions. Note that we're still separating the validation time-period.

# +
with torch.no_grad():
    pred_4step = kf_pm_univariate(
        dataset_pm_univariate.tensors[0],
        start_offsets=dataset_pm_univariate.start_datetimes,
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
    start_offsets=dataset_pm_multivariate_train.start_datetimes
)
# -
# We can generate our 4-step-ahead predictions for validation as we did before:

with torch.no_grad():
    pred_4step = kf_pm_multivariate(
        dataset_pm_multivariate.tensors[0],
        start_offsets=dataset_pm_multivariate.start_datetimes,
        n_step=4
    )
pred_4step.means.shape

# At this point, though, we run into a problem: we we have forecasts for both PM2.5 and PM10, but we ultimately want a forecast for their *sum*. With untransformed data, we could take advantage of the fact that [sum of correlated normals is still normal](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables#Correlated_random_variables):

torch.sum(pred_4step.means, 2)

# In our case this unfortunately won't work: we have log-transformed our measures. This seems like it was the right choice (i.e. our residuals look reasonably normal and i.i.d):

pred_4step.plot(pred_4step.to_dataframe(dataset_pm_multivariate, type='components').query("process=='residuals'"))

# In this case, we **can't take the sum of our forecasts to get the forecast of the sum**, and [there's no simple closed-form expression for the sum of lognormals](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C14&q=SUMS+OF+LOGNORMALS&btnG=).
#
# One option that is fairly easy in `torchcast` is to use a [Monte-Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) approach: we'll just generate random-samples based on the means and covariances underlying our forecast. In that case, the sum of the PM2.5 + PM10 forecasted-samples *is* the forecasted PM sum we are looking for:

# +
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

# We see that this approach has reduced our error: substantially in the training period, and moderately in the validation period. We can look at the per-site differences to reduce common sources of noise and see that the reduction is consistent (it holds for all but one site):

df_multivariate_error.\
    merge(df_univariate_error, on=['station', 'validation']).\
    assign(error_diff = lambda df: df['error_x'] - df['error_y']).\
    boxplot('error_diff', by='validation')


