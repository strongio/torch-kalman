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

from torchcast.utils.datasets import load_air_quality_data
from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LocalTrend, Season
from torchcast.utils.data import TimeSeriesDataset

import numpy as np

np.random.seed(2021-1-21)
torch.manual_seed(2021-1-21)
# -

# # Quick Start
#
# `torchcast` is a python package for time-series forecasting in PyTorch. Its focus is on training and forecasting with *batches* of time-serieses, rather than training separate models for one time-series at a time. In addition, it provides robust support for *multivariate* time-series, where multiple correlated measures are being forecasted.
#
# To briefly provide an overview of these features, we'll use a dataset from the [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data). It includes data on air pollutants and weather from 12 sites.

# + {"tags": ["remove_cell"]}
df_aq = load_air_quality_data('weekly')

df_aq

# + [markdown] {"hidePrompt": true}
# ### Prepare our Dataset
#
# In `torchcast` we set up our data and model with the following:
#
# - The `groups` which define separate time-serieses. Here we have multiple sites. Groups are not necessarily simultanoues to each other (e.g. we could have time-series of product purchases with products having varying release-dates) and correlations across these groups are not modeled.
# - The `measures` which define separate metrics we are measuring simultanously. Here we have the two kinds of particulate-matter (2.5 and 10).
#
# The `TimeSeriesDataset` is similar to PyTorch's native `TensorDataset`, with some useful metadata on the batch of time-serieses (the station names, the dates for each).
#
# For a quick example, we'll focus on predicting particulate-matter (PM2.5 and PM10). We'll log-transform since this is strictly positive.

# +
df_aq['PM2p5_log10'] = np.log10(df_aq['PM2p5'])
df_aq['PM10_log10'] = np.log10(df_aq['PM10'])

# create a dataset:
dataset_all = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq,
    dt_unit='W',
    measure_colnames=['PM2p5_log10', 'PM10_log10'],
    group_colname='station', 
    time_colname='week'
)

# Split out training period:
SPLIT_DT = np.datetime64('2016-02-22') 
dataset_train, _ = dataset_all.train_val_split(dt=SPLIT_DT)
dataset_train
# -

# ### Specify our Model
#
# In `torchcast` our forecasting model is defined by `measures` and `processes`. The `processes` give rise to the measure-able behavior. Here we'll specify a random-walk/trend component and a yearly seasonal component for each pollutant.

processes = []
for m in dataset_train.measures[0]:
    processes.extend([
        LocalTrend(id=f'{m}_trend', measure=m),
        Season(id=f'{m}_day_in_year', period=365.25 / 7, dt_unit='W', K=4, measure=m)
    ])
kf_first = KalmanFilter(measures=dataset_train.measures[0], processes=processes)

# ### Train our Model
#
# The `KalmanFilter` class provides a convenient `fit()` method that's useful for avoiding standard boilerplate for full-batch training:

# %%time
kf_first.fit(
    dataset_train.tensors[0], 
    start_datetimes=dataset_train.start_datetimes, 
    tol=.01
)

# Calling `forward()` on our `KalmanFilter` produces a `Predictions` object. If you're writing your own training loop, you'd simply use the `log_prob()` method as the loss function:

# +
pred = kf_first(
        dataset_train.tensors[0], 
        start_datetimes=dataset_train.start_datetimes,
        out_timesteps=dataset_all.tensors[0].shape[1]
)

loss = -pred.log_prob(dataset_train.tensors[0]).mean()
print(loss)
# -

# ### Inspect & Visualize our Output

# `Predictions` can easily be converted to Pandas `DataFrames` for ease of inspecting predictions, comparing them to actuals, and visualizing:

df_pred = pred.to_dataframe(dataset_all)
# bias-correction for log-transform (see https://otexts.com/fpp2/transformations.html#bias-adjustments)
df_pred['mean'] += .5 * (df_pred['upper'] - df_pred['lower']) / 1.96 ** 2
# inverse the log10:
df_pred[['actual','mean','upper','lower']] = 10 ** df_pred[['actual','mean','upper','lower']]
df_pred

df_pred['percent_error'] = np.abs(df_pred['mean'] - df_pred['actual']) / df_pred['actual']
print("Percent Error: {:.1%}".format(df_pred.query("time>@SPLIT_DT")['percent_error'].mean()))

# The `Predictions` class comes with a `plot` classmethod for getting simple plots of forecasted vs. actual:

print(pred.plot(df_pred.query("group=='Changping'"), split_dt=SPLIT_DT))

# Finally you can produce dataframes that decompose the predictions into the underlying `processes` that produced them:

pred.plot(pred.to_dataframe(dataset_all, type='components').query("group=='Changping'"), split_dt=SPLIT_DT)

# +
# TODO: links to more
