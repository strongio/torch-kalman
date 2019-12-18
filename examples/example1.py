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

import torch
from torch.optim import LBFGS
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, Season, FourierSeasonDynamic, LinearModel

import numpy as np
import pandas as pd
from torch_kalman.utils.data import TimeSeriesDataset

season_config = {
    'season_start': pd.Timestamp('2007-01-01'),  # arbitrary monday at midnight
    'dt_unit': 'D'  # <--- depends on sim
}

# TODO:
df_train = None
df_val = None
predictors = ['X1', 'X2']

kf = KalmanFilter(
    measures=['y'],
    processes=[
        LocalLevel(id='local_level').add_measure('y'),
        Season(id='day_in_week', seasonal_period=7, **season_config).add_measure('y'),
        FourierSeasonDynamic(
            id='day_in_month', seasonal_period=30, K=2, **season_config
        ).add_measure('y'),
        LinearModel(id='lm', covariates=predictors)
    ]
)
kf.opt = LBFGS(kf.parameters())

train_batch = TimeSeriesDataset.from_dataframe(dataframe=df_train, dt_unit=season_config['dt_unit'])
train_batch = train_batch.split_measures(['y'], predictors)
# remove nan-padding for predictors only, assumes predictors are centered:
train_batch.tensors[1][torch.isnan(train_batch.tensors[1])] = 0.0


def closure():
    kf.opt.zero_grad()
    y, X = train_batch.tensors
    pred = kf(y, start_datetimes=train_batch.start_datetimes, predictors=X)
    loss = -pred.log_prob(y).mean()
    loss.backward()
    return loss


NUM_EPOCHS = 100
for epoch in range(NUM_EPOCHS):
    loss = kf.opt.step(closure)
    print(f"EPOCH {epoch}, LOSS {loss.item()}")


