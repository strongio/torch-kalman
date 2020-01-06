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

# ## Dependencies

# +
import numpy as np
import pandas as pd

import requests
from io import BytesIO 
from zipfile import ZipFile

import math
from numpy.random import normal
import datetime
from sklearn.preprocessing import StandardScaler

import torch
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, LocalTrend, Season, FourierSeason, FourierSeasonFixed, NN
from torch_kalman.utils.data import TimeSeriesDataset, TimeSeriesDataLoader

from plotnine import *
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose, STL
# -

# ## UCI Machine Learning Data Repository - Beijing Multi-Site Air-Quality Data Set

# This data set includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites. The air-quality data are from the Beijing Municipal Environmental Monitoring Center. The meteorological data in each air-quality site are matched with the nearest weather station from the China Meteorological Administration. The time period is from March 1st, 2013 to February 28th, 2017. 
#
# For this example we will focus on 1 site (Tiantan). We will predict the amount of carbon monoxide pollutant present in the air (ug/m^3) at the end of each work day (5p.m. local time)

# Read in data
try:
    df = pd.read_csv("./PRSA2017_Data_20130301-20170228.csv")
except FileNotFoundError:
    response = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip')
    zip_file = ZipFile(BytesIO(response.content))
    files = zip_file.namelist()
    df = pd.read_csv(zip_file.open(files[10]))

# For now we're only interested in CO levels right at the end of the work day:
df = df[df.hour == 17]

# #### Understanding the data

df.dtypes

df.describe()

df.isnull().sum()

np.unique(df['station'], return_counts=True)

np.unique(df['wd'].astype(str), return_counts=True)

# #### Data transformations

# +
# Create time index
df['time'] = pd.to_datetime(df[['year','month','day']])
#df['time'] = pd.to_datetime(df[['year','month','day','hour']])

# Dummy code Wind Direction (wd)
df = pd.concat([df, pd.get_dummies(df['wd'])], axis=1)
df.drop(['wd'], inplace=True, axis=1)

# Linearly interpolate any missing values in CO and predictors:
for col in ['CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']:
    df[col] = df[col].interpolate(method='linear')

# Training data: pre- March 2016
# Testing data: March 2016 - March 2017
train_test_cutoff_date = pd.Timestamp("2016-3-1")
train_df = df[df['time'] < train_test_cutoff_date].reset_index(drop=True)
test_df = df[df['time'] >= train_test_cutoff_date].reset_index(drop=True)

# Standardize data: Fit and transform training data
train_scaler = StandardScaler()
train_df[['CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'E', 'ENE', 'ESE', 'N', 'NE', 'NNE',
       'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']] = train_scaler. \
fit_transform(train_df[['CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'E', 'ENE', 'ESE', 'N', 'NE', 'NNE',
       'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']].to_numpy())

# Standardize data: Transform testing data using train_scaler
test_df[['CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'E', 'ENE', 'ESE', 'N', 'NE', 'NNE',
       'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']] = train_scaler. \
transform(test_df[['CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'E', 'ENE', 'ESE', 'N', 'NE', 'NNE',
       'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']].to_numpy())
# -

# It appears our target variable has an annual seasonality:

print(
    ggplot(train_df, aes(x='time', y='CO')) + \
                geom_line() + \
                ylab("Scaled CO") + xlab("Time") + \
                theme_minimal() + theme(figure_size=(12, 6))
)

# `CO` Variance appears to be seasonal as well:

var_df = train_df.groupby(['year','month'])['CO'].var().reset_index().assign(day=1)
var_df['time'] = pd.to_datetime(var_df[['year','month','day']])
print(
    ggplot(var_df, aes(x='time', y='CO')) + \
                geom_line() + \
                ylab("Scaled CO Variance") + xlab("Time") + \
                theme_minimal() + theme(figure_size=(12, 6))
)

# The meteorological data (potential predictors) also appear to have an annual seasonality:

# +
tempdf = train_df[['time','CO','TEMP','PRES','RAIN']].melt(id_vars=['time'])

print(
    ggplot(tempdf, aes(x='time', y='value', color='variable')) + \
                geom_line() + \
                ylab("Amount") + xlab("Time") + \
                theme_minimal() + ggtitle('Scaled CO & predictors') + theme(figure_size=(12, 6))
)
# -

# #### Time series decomposition

result = seasonal_decompose(df['CO'], model="additive", period=365)
result.plot()
pyplot.show()

# ## Train on target series only

# Let's build our Kalman filter. First let's define some global variables that will help us deinfe when seasonalities start and at what time granularity the data is at:

# In roder to create our Kalman Filter, we need *measures* and *processes*. Measures are simply our outcome(s) of interest. Processes are models we use to understand our system- i.e. what makes us go from one state to another. In time series applications, we often already have an intuitive understanding of such processes. There could be a trend, different seasonalities, exogneous factors, etc. `torch-kalman` allows you to specify these in an easy and clear way. For instance we can define a `LocalLevel` (random walk) process as well as `FourierSeasonFixed` a seasonality process using a fourier series:

kf = KalmanFilter(
    measures=['CO'],
    processes=[
        LocalLevel(id='local_level').add_measure('CO'),
        FourierSeasonFixed(id='day_in_year', seasonal_period=365, K=2, dt_unit='D').add_measure('CO')
    ]
)

# Once this is in place, we can leverage `torch` to optimize our parameters of interest:
#
# - Measurement Noise
# - Process Noise
#

kf.opt = torch.optim.Adam(kf.parameters(), lr=0.1)

# And before training, we'll use `TimeSeriesDataset`s (which mimic `TensorDatasets`) to shape our data in a way that can be used by `torch-kalman`. This shape is `(group, time, measure)` but for now we only have 1 group (`station`) and 1 measure (`CO`):

# +
train_batch = TimeSeriesDataset.from_dataframe(dataframe=train_df, 
                                               group_colname = 'station',
                                               time_colname = 'time',
                                               measure_colnames = ['CO'],
                                               dt_unit='D')

test_batch = TimeSeriesDataset.from_dataframe(dataframe=test_df, 
                                               group_colname = 'station',
                                               time_colname = 'time',
                                               measure_colnames = ['CO'],
                                               dt_unit='D')


# -

# And finally we train our model:

# +
def closure():
    kf.opt.zero_grad()
    y = train_batch.tensors[0]
    pred = kf(y, 
              start_datetimes=train_batch.start_datetimes)
    loss = -pred.log_prob(y).mean()
    loss.backward()
    return loss

def valid():
    with torch.no_grad():
        y = test_batch.tensors[0]
        pred = kf(y, 
                  start_datetimes=test_batch.start_datetimes)
        loss = -pred.log_prob(y).mean()
        return loss

train_losses = []
valid_losses = []
NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    train_loss = kf.opt.step(closure)
    valid_loss = valid()
    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss.item())
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss.item()}, VALIDATION LOSS {valid_loss.item()}")
    
pyplot.plot(train_losses)
pyplot.plot(valid_losses)
pyplot.show()
# -

# Let's view our performance. We'll pass our training data, and forecast out the length of our validation data. We then merge our predictions with all data and plot.

# +
all_df = pd.concat([train_df,test_df])
all_batch = TimeSeriesDataset.from_dataframe(dataframe=all_df, 
                                               group_colname = 'station',
                                               time_colname = 'time',
                                               measure_colnames = ['CO'],
                                               dt_unit='D')

# get validation forecasts
with torch.no_grad():

    trainy = train_batch.tensors[0]
    ally = all_batch.tensors[0]

    preds = kf(trainy, 
               start_datetimes = train_batch.start_datetimes,
               out_timesteps = ally.shape[1])

# join validation predictions with all data



pred_df = preds.to_dataframe(all_batch) \
.assign(predicted_min = lambda x: x['predicted_mean'] - x['predicted_std'],
        predicted_max = lambda x: x['predicted_mean'] + x['predicted_std']).\
rename(columns={'actual':'CO'})
# -

# Our average predictions appear to follow the yearly trend, and our uncertainty increase the farther out we make forecasts:

# plot
print(
    ggplot(pred_df,
           aes(x='time')) +
    geom_line(aes(y='CO'), color='blue', size=1) +
    geom_line(aes(y='predicted_mean'), color='red', size=1, linetype = 'dashed') +
    geom_ribbon(aes(ymin='predicted_min', ymax='predicted_max'), alpha=.20) +
    theme_minimal() +
    theme(figure_size=(12, 6)) +
    geom_vline(xintercept=train_test_cutoff_date) +
    scale_x_date(name="") + scale_y_continuous(name="Scaled CO")
)

# Accuracy statistics:

test_pred_df = pred_df[pred_df['time'] >= pd.Timestamp(train_test_cutoff_date)]
rmse = np.sqrt(((test_pred_df['CO'] - test_pred_df['predicted_mean']) ** 2).mean())
mae = np.mean(np.abs((test_pred_df['CO'] - test_pred_df['predicted_mean'])))
print(f"VALIDATION RMSE {rmse}, VALIDATION MAE {mae}")

# ## Train with exogenous predictors that are embedded in a single state value using NN

# Our predictions look ok, but can we leverage additional features like the meteorlogical data to improve accuracy?

predictors = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

# Since we're using `torch` why not leverage some sort of architecture to take predictors in as inputs, and output some latent representation of the process state?

input_dim = len(predictors)
output_dim = 1
state_emb_model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, 50, bias=False),
    torch.nn.Tanh(),
    torch.nn.Linear(50, output_dim, bias=False)
)

# We can specify these exogenous predictors as simply another process:

kf_exo = KalmanFilter(
    measures=['CO'],
    processes=[
        LocalLevel(id='local_level').add_measure('CO'),
        FourierSeasonFixed(id='day_in_year', seasonal_period=365, K=2, dt_unit='D').add_measure('CO'),
        NN(id='predictors',
           input_dim=input_dim,
           state_dim=output_dim,
           nn_module=state_emb_model,
           add_module_params_to_process=False).add_measure('CO')
    ]
)

# And now we train the same as before:

# +
kf_exo.opt = torch.optim.Adam([{'params': kf_exo.parameters(), 'lr': 0.1},
                                  {'params': state_emb_model.parameters(), 'lr': 0.01}])

train_batch = TimeSeriesDataset.from_dataframe(dataframe=train_df, 
                                               group_colname = 'station',
                                               time_colname = 'time',
                                               y_colnames=['CO'],
                                               X_colnames=predictors,
                                               dt_unit='D')

test_batch = TimeSeriesDataset.from_dataframe(dataframe=test_df, 
                                               group_colname = 'station',
                                               time_colname = 'time',
                                               y_colnames=['CO'],
                                               X_colnames=predictors,
                                               dt_unit='D')


def closure():
    kf_exo.opt.zero_grad()
    y, X = train_batch.tensors
    pred = kf_exo(y, 
                  start_datetimes = train_batch.start_datetimes,
                  predictors = X)
    loss = -pred.log_prob(y).mean()
    loss.backward()
    return loss

def valid():
    with torch.no_grad():
        y, X = test_batch.tensors
        pred = kf_exo(y, 
                      start_datetimes=test_batch.start_datetimes,
                      predictors = X)
        loss = -pred.log_prob(y).mean()
        return loss

train_losses = []
valid_losses = []
NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    train_loss = kf_exo.opt.step(closure)
    valid_loss = valid()
    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss.item())
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss.item()}, VALIDATION LOSS {valid_loss.item()}")
    
pyplot.plot(train_losses)
pyplot.plot(valid_losses)
pyplot.show()
# -

# It looks like our accuracy improved:

# +
all_df = pd.concat([train_df,test_df])
all_batch = TimeSeriesDataset.from_dataframe(dataframe=all_df, 
                                               group_colname = 'station',
                                               time_colname = 'time',
                                               y_colnames=['CO'],
                                               X_colnames=predictors,
                                               dt_unit='D')

with torch.no_grad():
    trainy, _ = train_batch.tensors
    _, allX = all_batch.tensors
    preds = kf_exo(trainy, 
                   start_datetimes = train_batch.start_datetimes,
                   predictors = allX,
                   out_timesteps = allX.shape[1])

pred_df = preds.to_dataframe(all_batch) \
.assign(predicted_min = lambda x: x['predicted_mean'] - x['predicted_std'],
        predicted_max = lambda x: x['predicted_mean'] + x['predicted_std']).\
rename(columns={'actual':'CO'})

print(
    ggplot(pred_df,
           aes(x='time')) +
    geom_line(aes(y='CO'), color='blue', size=1) +
    geom_line(aes(y='predicted_mean'), color='red', size=1, linetype = 'dashed') +
    geom_ribbon(aes(ymin='predicted_min', ymax='predicted_max'), alpha=.20) +
    theme_minimal() +
    theme(figure_size=(12, 6)) +
    geom_vline(xintercept=train_test_cutoff_date) +
    scale_x_date(name="") + scale_y_continuous(name="Scaled CO")
)
# -

# It did!

test_pred_df = pred_df[pred_df['time'] >= pd.Timestamp(train_test_cutoff_date)]
rmse = np.sqrt(((test_pred_df['CO'] - test_pred_df['predicted_mean']) ** 2).mean())
mae = np.mean(np.abs((test_pred_df['CO'] - test_pred_df['predicted_mean'])))
print(f"VALIDATION RMSE {rmse}, VALIDATION MAE {mae}")

# #### possible improvements
#
# - log transform
# - train in cross validated way
# - use more predictors / create fourier terms of predictors

# #### next example ideas:
#
# - multiple groups / measures
# - analyze uncertainty further, plot many forecasted series based on sampling gaussin at each point
