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

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

np.random.seed(2020-1-3)
torch.manual_seed(2020-1-3)
# -

# ## UCI Machine Learning Data Repository - Beijing Multi-Site Air-Quality Data Set
#
# This dataset includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites. The air-quality data are from the Beijing Municipal Environmental Monitoring Center. 
#
# The meteorological data in each air-quality site are matched with the nearest weather station from the China Meteorological Administration. The time period is from March 1st, 2013 to February 28th, 2017. 
#
# In addition to time/station information, the dataset consists of the following columns:
#
# - PM2.5: PM2.5 concentration (ug/m^3)
# - PM10: PM10 concentration (ug/m^3)
# - SO2: SO2 concentration (ug/m^3)
# - NO2: NO2 concentration (ug/m^3)
# - CO: CO concentration (ug/m^3)
# - O3: O3 concentration (ug/m^3)
# - TEMP: temperature (degree Celsius)
# - PRES: pressure (hPa)
# - DEWP: dew point temperature (degree Celsius)
# - RAIN: precipitation (mm)
# - wd: wind direction
# - WSPM: wind speed (m/s)

# +
import requests
from zipfile import ZipFile
from io import BytesIO

# Read in data
try:
    df_aq = pd.read_csv("./PRSA2017_Data_20130301-20170228.csv")
except FileNotFoundError:
    response =\
        requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip')
    zip_file = ZipFile(BytesIO(response.content))
    files = zip_file.namelist()
    df_aq = pd.concat([pd.read_csv(zip_file.open(f)) for f in files if f.endswith('csv')])
    df_aq.to_csv("./PRSA2017_Data_20130301-20170228.csv", index=False)

del df_aq['No']
df_aq['time'] = pd.to_datetime(df_aq.loc[:,['year','month','day','hour']])    
# -

# For now, we'll focus on generating daily forecasts. We'll downsample by taking the values from the end of the workday.

df_aq_daily = df_aq.\
    loc[df_aq['hour'] == 17,:].\
    drop(columns=['hour']).\
    reset_index(drop=True).\
    assign(date = lambda df: df.pop('time').astype('datetime64[D]'))
df_aq_daily

# We'll focus on predicting CO levels. 

df_aq_daily.query("station=='Aotizhongxin'").plot('date','CO', figsize=(15,3))
df_aq_daily.query("station=='Aotizhongxin'").plot('date','CO', figsize=(15,3),logy=True)

# This predictor is strictly positive and its variance is variable throughout the year, so we'll use a log-transform. We'll also scale the data so that it's centered around zero on the log scale.

df_aq_daily['CO_log_scaled'] = np.log10(df_aq_daily['CO'] / df_aq_daily['CO'].mean())

# ## Single Station
#
# As a first example, let's generate predictions for a single station.
#
# We'll first create a `TimeSeriesDataset` object. This behaves similarly to pytorch's `TensorDataset`, but contains useful metadata about the tensors relating to:
#
# - The groups (stations) in the dataset, indexed in the tensor's first dimension.
# - The times in the dataset, indexed in the tensor's second dimension.
# - The measures in the dataset, indexed in the tensor's third dimension.
#
# For now we have one group (Aotizhongxin) and one measure (log-transformed CO-levels).

dataset_aoti = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq_daily.query("station=='Aotizhongxin'"), 
    dt_unit='D',
    group_colname='station',
    time_colname='date',
    measure_colnames=['CO_log_scaled']
)

SPLIT_DT = np.datetime64('2016-03-01')
dataset_aoti_train, dataset_aoti_val = dataset_aoti.train_val_split()
dataset_aoti_train, dataset_aoti_val

# In order to create our Kalman Filter, we need *measures* and *processes*. 
#
# Measures are simply our outcome(s) of interest. 
#
# Processes are models we use to understand our system -- i.e. what makes the system transition from one state to another, and how this latent system can be translated into the observable *measures*. 
#
# In time series applications, we often already have an intuitive understanding of such processes. There could be a trend, different seasonalities, exogneous factors, etc. `torch-kalman` comes with several flexible processes. Here, we will define a `LocalLevel` (random walk) process as well as `FourierSeason` process that captures seasonality using a fourier-series.

kf_aoti = KalmanFilter(
    measures=['CO_log_scaled'],
    processes=[
        LocalLevel(id='local_level').add_measure('CO_log_scaled'),
        FourierSeason(id='day_in_year', seasonal_period=365.25, dt_unit='D', K=2, fixed=True).add_measure('CO_log_scaled')
    ]
)

# Since our dataset is fairly small, we'll use full-batch LBFGS.
#
# The output of the forward pass is a `StateBeliefOverTime` object. These represent the predictions (by default one-step-ahead) generated by the kalman-filter. These predictions come with a mean and covariance, and so can be evaluated against the actual data using a (negative) log-probability critierion:

# +
kf_aoti.opt = LBFGS(kf_aoti.parameters(), lr=.25, max_eval=10)

def closure():
    kf_aoti.opt.zero_grad()
    pred = kf_aoti(dataset_aoti_train.tensors[0], start_datetimes=dataset_aoti_train.start_datetimes)
    loss = -pred.log_prob(dataset_aoti_train.tensors[0]).mean()
    loss.backward()
    return loss


# -

for epoch in range(10):
    train_loss = kf_aoti.opt.step(closure).item()
    with torch.no_grad():
        pred = kf_aoti(dataset_aoti_val.tensors[0], start_datetimes=dataset_aoti_val.start_datetimes)
        val_loss = -pred.log_prob(dataset_aoti_val.tensors[0]).mean().item()
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss}, VAL LOSS {val_loss}")

# The `StateBeliefOverTime` also has `to_dataframe()` and `plot()` methods, which are useful for visualizing our model's forecasts.

pred = kf_aoti(
    dataset_aoti_train.tensors[0], 
    start_datetimes=dataset_aoti_train.start_datetimes,
    out_timesteps=dataset_aoti.tensors[0].shape[1]
)
pred.plot(pred.to_dataframe(dataset_aoti), split_dt=SPLIT_DT)

# We can also get a dataframe (and plot) of the components of the model, which are the measureable parts of the processes. 

pred.plot(pred.to_dataframe(dataset_aoti, type='components'), split_dt=SPLIT_DT)

# TODO: MAPE

# ## Multi-Station
#
# `torch-kalman` facilitates training a model on multiple time-series. This lets the model share hard-to-learn parameters (like process and measure covariances) across groups, while tailoring predictions to each specific group (e.g. each group might have a different seasonal pattern). 
#
# For simplicity the below still uses full-batch training. If you need to split your data into batches see `TimeSeriesDataLoader`.

# +
dataset_all = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq_daily,
    dt_unit='D',
    group_colname='station',
    time_colname='date',
    measure_colnames=['CO_log_scaled']
)

dataset_train, dataset_val = dataset_all.train_val_split()
dataset_train, dataset_val

# +
kf_all = KalmanFilter(
    measures=['CO_log_scaled'],
    processes=[
        LocalLevel(id='local_level').add_measure('CO_log_scaled'),
        FourierSeason(id='day_in_year', seasonal_period=365.25, dt_unit='D', K=2, fixed=True).add_measure('CO_log_scaled')
    ]
)

kf_all.opt = LBFGS(kf_all.parameters(), lr=.25, max_eval=10)

def closure():
    kf_all.opt.zero_grad()
    pred = kf_all(dataset_train.tensors[0], start_datetimes=dataset_train.start_datetimes)
    loss = -pred.log_prob(dataset_train.tensors[0]).mean()
    loss.backward()
    return loss


# -

for epoch in range(10):
    train_loss = kf_all.opt.step(closure).item()
    with torch.no_grad():
        pred = kf_all(dataset_val.tensors[0], start_datetimes=dataset_val.start_datetimes)
        val_loss = -pred.log_prob(dataset_val.tensors[0]).mean().item()
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss}, VAL LOSS {val_loss}")

pred = kf_all(
    dataset_train.tensors[0], 
    start_datetimes=dataset_train.start_datetimes,
    out_timesteps=dataset_all.tensors[0].shape[1]
)

# TODO: MAPE

# ## With Exogenous Predictors
#
# In some contexts, we want to use exogenous predictors. For example, we may only be interested in one-step-ahead forecasts, and we can use 1-day-lagged predictors. Or we may be interested in [estimating counterfactuals](https://google.github.io/CausalImpact/CausalImpact.html).

predictors_raw = ['TEMP', 'PRES', 'DEWP', 'WSPM', 'RAIN']
for col in predictors_raw:
    df_aq_daily.query("station=='Aotizhongxin'").plot('date', col, figsize=(15,3))

# First, we'll center and scale our exogenous predictors, which is helpful for speeding up optimization:

# +
df_predictors_daily = df_aq_daily.\
    loc[:,['station', 'date', 'CO_log_scaled'] + predictors_raw].\
    assign(RAIN_log1p = lambda df: np.log1p(df['RAIN']),
           WSPM_log1p = lambda df: np.log1p(df['WSPM']))

predictors = ['TEMP', 'PRES', 'DEWP', 'WSPM_log1p', 'RAIN_log1p']

pred_scaler = StandardScaler().fit(df_predictors_daily.loc[df_predictors_daily['date'] < SPLIT_DT, predictors])
df_predictors_daily[predictors] = pred_scaler.transform(df_predictors_daily[predictors])
# -

# Since a tensor of exogenous predictors is such a common use-case, `TimeSeriesDataset` supports passing `y_colnames` and `X_colnames` separately, instead of passing `measure_colnames`.
#
# This gives us a TimeSeriesDataset with two tensors. For predictors, we need to impute missing values.

# +
dataset_pred = TimeSeriesDataset.from_dataframe(
    dataframe=df_predictors_daily,
    dt_unit='D',
    group_colname='station',
    time_colname='date',
    y_colnames=['CO_log_scaled'],
    X_colnames=predictors
)

# impute nans (since standardized, imputing w/zeros means imputing w/mean)
_, X = dataset_pred.tensors
X[torch.isnan(X)] = 0.0

dataset_train, dataset_val = dataset_pred.train_val_split()
dataset_train, dataset_val

# since realigned, creates new nan-padding:
_, X = dataset_train.tensors
X[torch.isnan(X)] = 0.0
_, X = dataset_val.tensors
X[torch.isnan(X)] = 0.0
# -

# We add these predictors by adding a `LinearModel` process to our model. 

# +
kf_pred = KalmanFilter(
    measures=['CO_log_scaled'],
    processes=[
        LocalLevel(id='local_level').add_measure('CO_log_scaled'),
        FourierSeason(id='day_in_year', seasonal_period=365.25, dt_unit='D', K=2, fixed=True).add_measure('CO_log_scaled'),
        LinearModel(id='predictors', covariates=predictors).add_measure('CO_log_scaled')
    ]
)

kf_pred.opt = LBFGS(kf_pred.parameters(), lr=.25, max_eval=10)

def closure():
    kf_pred.opt.zero_grad()
    y, X = dataset_train.tensors
    pred = kf_pred(y, predictors=X, start_datetimes=dataset_train.start_datetimes)
    loss = -pred.log_prob(y).mean()
    loss.backward()
    return loss


# -

for epoch in range(10):
    train_loss = kf_pred.opt.step(closure).item()
    y, X = dataset_val.tensors
    with torch.no_grad():
        pred = kf_pred(y, predictors=X, start_datetimes=dataset_val.start_datetimes)
        val_loss = -pred.log_prob(y).mean().item()
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss}, VAL LOSS {val_loss}")

# Plotting our predictions and components suggests that the predictors are doing a lot of the work previously done by `LocalLevel` and `FourierSeason`.

# +
y, _ = dataset_train.tensors
_, X = dataset_pred.tensors
pred = kf_pred(
    y,
    predictors=X,
    start_datetimes=dataset_train.start_datetimes,
    out_timesteps=X.shape[1]
)

print(pred.plot(pred.to_dataframe(dataset_pred).query("group=='Aotizhongxin'"), split_dt=SPLIT_DT))

df_components = pred.to_dataframe(dataset_pred, type='components')

print(pred.plot(df_components.query("group=='Aotizhongxin'"), split_dt=SPLIT_DT))
# -
# If we plot just the `predictors` process, then we can get a legend, helping us understand what each predictor contributes to the forecast:

print(pred.plot(df_components.query("process == 'predictors'"), max_num_groups=2, split_dt=SPLIT_DT))
