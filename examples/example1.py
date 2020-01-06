# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline

import torch
from torch.optim import LBFGS

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, Season, FourierSeason, LinearModel
from torch_kalman.utils.data import TimeSeriesDataset
from torch_kalman.utils.simulate import simulate_daily_series

import numpy as np
import pandas as pd

np.random.seed(2020-1-3)
torch.manual_seed(2020-1-3)
# -



data = simulate_daily_series(20, 180)
data.head()

predictors = ['X1','X2']
SPLIT_DT = np.datetime64('2007-05-01')

# +
dataset_all = TimeSeriesDataset.from_dataframe(
    dataframe=data, 
    dt_unit='D',
    group_colname='group',
    time_colname='time',
    y_colnames=['y'],
    X_colnames=predictors
)

dataset_train, dataset_val = dataset_all.train_val_split(dt=SPLIT_DT)
# -

# ## Create our Kalman Filter

kf = KalmanFilter(
    measures=['y'],
    processes=[
        LocalLevel(id='local_level').add_measure('y'),
        Season(id='day_in_week', seasonal_period=7, dt_unit='D').add_measure('y'),
        LinearModel(id='linear_model', covariates=predictors).add_measure('y')
    ]
)

# Since it's a small dataset, we'll train using full-batch LBFGS with early-stopping.

# +
kf.opt = LBFGS(kf.parameters(), lr=.1)

def closure():
    kf.opt.zero_grad()
    y, X = dataset_train.tensors
    pred = kf(y, start_datetimes=dataset_train.start_datetimes, predictors=X)
    loss = -pred.log_prob(y).mean()
    if loss.requires_grad:
        loss.backward()
    return loss


# -

prev_loss = float('nan')
for epoch in range(100):
    loss = kf.opt.step(closure).item()
    delta = prev_loss - loss
    print(f"EPOCH {epoch}, LOSS {loss}, DELTA {delta}")
    if abs(delta) < .001:
        break
    prev_loss = loss

# ## Visualize Predictions

dataset_train_forecast, _ = dataset_all.train_val_mask(dt=SPLIT_DT)

y, X = dataset_train_forecast.tensors
pred = kf(
    y, 
    start_datetimes=dataset_train.start_datetimes, 
    predictors=X,
    out_timesteps=X.shape[1]
)

df_predictions = pred.to_dataframe(dataset_all)
df_predictions.head()

print(pred.plot(df_predictions.query("group == 1"), split_dt=SPLIT_DT))

# ## Visualize Forecast-Components

df_components = pred.to_dataframe(dataset_all, type='components')
df_components.head()

print(pred.plot(df_components.query("group == 1"), split_dt=SPLIT_DT))


