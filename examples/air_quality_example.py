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
from torch_kalman.process import LocalLevel, LocalTrend, Season, FourierSeason, LinearModel
from torch_kalman.utils.data import TimeSeriesDataset
from torch_kalman.utils.simulate import simulate_daily_series

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

np.random.seed(2020-1-3)
torch.manual_seed(2020-1-3)
# -

# # Introduction to Torch-Kalman

# ## Beijing Multi-Site Air-Quality Dataset
#
# This dataset comes from the [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data). It includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites. The air-quality data are from the Beijing Municipal Environmental Monitoring Center. 
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
df_aq = df_aq.rename(columns={'PM2.5':'PM2p5'})
# -

SPLIT_DT = np.datetime64('2016-02-22')

# To simplify the example, we'll focus on weekly averages for two measures: PM10 and SO2. Since these measures are strictly positive, we log-transform them.

# +
df_aq_weekly = df_aq.\
    assign(date = lambda df: df['time'].astype('datetime64[D]') - pd.to_timedelta(df['time'].dt.dayofweek, unit='d')).\
    drop(columns = ['year','month','day','hour']).\
    groupby(['date','station']).\
    agg('mean').\
    reset_index()

col_means = df_aq_weekly.loc[df_aq_weekly['date'] < SPLIT_DT,:].mean()
col_stds = df_aq_weekly.loc[df_aq_weekly['date'] < SPLIT_DT,:].std()

measures = ['SO2','PM10']
measures_pp = [m + '_log10_scaled' for m in measures]

df_aq_weekly[measures_pp] = np.log10(df_aq_weekly[measures] / col_means[measures])


df_aq_weekly
# -

# The `TimeSeriesDataset` is similar to PyTorch's native `TensorDataset`, but includes metadata about the time-serieses that are in the dataset, such as the name for each (here, the station-name), the timestamps for each datapoint, and the measure-names.

# +
dataset_all = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq_weekly,
    dt_unit='W',
    measure_colnames=measures_pp,
    group_colname='station', 
    time_colname='date'
)

dataset_train, dataset_val = dataset_all.train_val_split(dt=SPLIT_DT)
dataset_train, dataset_val
# -

# In `torch-kalman` we think of a model in terms of its *processes*. Processes are how we describe what makes the system transition from one state to another, and how this latent system can be translated into the observable *measures*. 
#
# In time series applications, we often already have an intuitive understanding of such processes. There could be a trend, different seasonalities, exogneous factors, etc. `torch-kalman` comes with several flexible processes. Here, we will specify three processes:
#
# - The `LocalTrend` process captures long-range trends.
# - The `LocalLevel` process captures short-range changes that tend to decay back to zero.
# - The `FourierSeason` captures yearly seasonalities, using a [fourier-series](https://en.wikipedia.org/wiki/Fourier_series)
#
# Additionally, below shows off a few useful features of `torch-kalman`:
#
# - We are training on a multivarite time-series: that is, our time-series has two measures (SO2 and PM10) and our model will capture correlations across these.
# - We are going to train on, and predictor for, multiple time-serieses (i.e. multiple stations) at once. 
# - We are allowing the measure-variance -- the noise in the measures that cannot be captured by our processes -- to vary with the seasons, by passing 'seasonal' alias to `measure_var_predict`. (The `measure_var_predict` argument takes any `torch.nn.Module` that can be used for prediction, but 'seasonal' is an alias that tells the `KalmanFilter` to use a seasonal NN.)

# ## Our First Kalman Filter

processes = []
for measure in measures_pp:
    processes.extend([
        LocalTrend(
            id=f'{measure}_trend', multi=.01
        ).add_measure(measure),
        LocalLevel(
            id=f'{measure}_local_level',
            decay=(.90,1.00)
        ).add_measure(measure),
        FourierSeason(
            id=f'{measure}_day_in_year', seasonal_period=365.25 / 7., dt_unit='W', K=2, fixed=True
        ).add_measure(measure)
    ])
kf_first = KalmanFilter(measures=measures_pp, 
                      processes=processes, 
                      measure_var_predict=('seasonal',dict(K=2,period='yearly',dt_unit='W')))

# Since our dataset is fairly small, we'll use full-batch LBFGS.
#
# The output of the forward pass is a `StateBeliefOverTime` object. These represent the predictions (by default one-step-ahead) generated by the kalman-filter. These predictions come with a mean and covariance, and so can be evaluated against the actual data using a (negative) log-probability critierion:

# +
kf_first.opt = LBFGS(kf_first.parameters(), lr=.25, max_eval=10)

def closure():
    kf_first.opt.zero_grad()
    pred = kf_first(
        dataset_train.tensors[0], 
        start_datetimes=dataset_train.start_datetimes, 
    )
    loss = -pred.log_prob(dataset_train.tensors[0]).mean()
    loss.backward()
    return loss


# -

for epoch in range(15):
    train_loss = kf_first.opt.step(closure).item()
    with torch.no_grad():
        pred = kf_first(
            dataset_val.tensors[0], 
            start_datetimes=dataset_val.start_datetimes
        )
        val_loss = -pred.log_prob(dataset_val.tensors[0]).mean().item()
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss}, VAL LOSS {val_loss}")


# The `StateBeliefOverTime` also has `to_dataframe()` and `plot()` methods, which are useful for visualizing our model's forecasts.

# +
def inverse_transform(df: pd.DataFrame, col_means: pd.Series) -> pd.DataFrame:
    # inverse log10 transform:
    df = df.\
        assign(
            measure=lambda df: df['measure'].str.replace('_log10_scaled',''),
            _std=lambda df: (df['upper'] - df['lower']) / 1.96,
            mean=lambda df: 10 ** ( df['mean'] + .5 * df.pop('_std') **2 ),
            lower=lambda df: 10 ** ( df['lower']  ),
            upper=lambda df: 10 ** ( df['upper']  ),
            actual=lambda df: 10 ** ( df['actual'] ),
        )
    
    # inverse scaling:
    for col in ['mean','lower','upper','actual']:
        df[col] *= df['measure'].map(col_means.to_dict())

    return df

pred = kf_first(
    dataset_train.tensors[0], 
    start_datetimes=dataset_train.start_datetimes,
    out_timesteps=dataset_all.tensors[0].shape[1]
)

df_pred_aoti = inverse_transform(pred.to_dataframe(dataset_all), col_means)

pred.plot(df_pred_aoti.query("group=='Changping'"), split_dt=SPLIT_DT)
# -

# We can also get a dataframe (and plot) of the components of the model, which are the measureable parts of the processes. 

pred.plot(pred.to_dataframe(dataset_all, type='components').query("group=='Changping'"), split_dt=SPLIT_DT)

# ## Including Predictors
#
# In some contexts, we want to use exogenous predictors. For example, we may only be interested in one-step-ahead forecasts, and we can use 1-day-lagged predictors. Or we may be interested in [estimating counterfactuals](https://google.github.io/CausalImpact/CausalImpact.html).
#
# Here, we'll use the weather to predictor our measures of interest.

# +
predictors = ['TEMP', 'PRES', 'DEWP']
predictors_pp = [x + '_scaled' for x in predictors]

df_aq_weekly[predictors_pp] = (df_aq_weekly[predictors] - col_means[predictors]) / col_stds[predictors]
# -

# Since a tensor of exogenous predictors is such a common use-case, `TimeSeriesDataset` supports passing `y_colnames` and `X_colnames` separately, instead of passing `measure_colnames`.
#
# This gives us a TimeSeriesDataset with two tensors. For predictors, we need to impute missing values.

# +
dataset_all = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq_weekly,
    dt_unit='W',
    group_colname='station',
    time_colname='date',
    y_colnames=measures_pp,
    X_colnames=predictors_pp
)

dataset_train, dataset_val = dataset_all.train_val_split(dt=SPLIT_DT)

# impute nans (since standardized, imputing w/zeros means imputing w/mean)
for _dataset in (dataset_all, dataset_train, dataset_val):
    _, X = _dataset.tensors
    X[torch.isnan(X)] = 0.0
# -

# We add these predictors by adding a `LinearModel` process to our model. 

# +
kf_pred = KalmanFilter(
    measures=measures_pp,
    processes=processes + [
        LinearModel(id=f'{m}_predictors', covariates=predictors_pp).add_measure(m) 
        for m in measures_pp
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

for epoch in range(20):
    train_loss = kf_pred.opt.step(closure).item()
    y, X = dataset_val.tensors
    with torch.no_grad():
        pred = kf_pred(y, predictors=X, start_datetimes=dataset_val.start_datetimes)
        val_loss = -pred.log_prob(y).mean().item()
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss}, VAL LOSS {val_loss}")

# +
y, _ = dataset_train.tensors
_, X = dataset_all.tensors
pred = kf_pred(
    y,
    predictors=X,
    start_datetimes=dataset_train.start_datetimes,
    out_timesteps=X.shape[1]
)

print(
    pred.plot(inverse_transform(pred.to_dataframe(dataset_all).query("group=='Changping'"), col_means),split_dt=SPLIT_DT)
)

df_components = pred.to_dataframe(dataset_all, type='components')

print(pred.plot(df_components.query("group=='Changping'"), split_dt=SPLIT_DT))
# -
# If we plot just the `predictors` process, then we can get a legend, helping us understand what each predictor contributes to the forecast:

print(pred.plot(df_components.query("(group=='Changping') & (process.str.endswith('predictors'))"), split_dt=SPLIT_DT))
