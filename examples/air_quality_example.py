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

# + {"tags": ["remove_cell"]}
# generate the README with 
# ```
# jupyter nbconvert --to markdown air_quality_example.ipynb --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' --output README --output-dir ../
# ```
# %matplotlib inline

import torch
from torch.optim import LBFGS

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, LocalTrend, FourierSeason, LinearModel
from torch_kalman.utils.data import TimeSeriesDataset

import numpy as np
import pandas as pd

np.random.seed(2020-2-7)
torch.manual_seed(2020-2-7);
# -

# # Torch-Kalman
#
# Time-series forecasting models using Kalman-filters in PyTorch.

# ## Installation
#
# ```
# pip install https://github.com/strongio/torch-kalman#egg=torch_kalman
# ```

# ## Example: Beijing Multi-Site Air-Quality Dataset
#
# This dataset comes from the [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data). It includes data on air pollutants and weather from 12 sites. To simplify the example, we'll focus on weekly averages for two measures: PM10 and SO2. Since these measures are strictly positive, we log-transform them.

# + {"tags": ["remove_cell"]}
# Read in data
try:
    df_aq = pd.read_csv("./PRSA2017_Data_20130301-20170228.csv")
except FileNotFoundError:
    import requests
    from zipfile import ZipFile
    from io import BytesIO
    response =\
        requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip')
    zip_file = ZipFile(BytesIO(response.content))
    files = zip_file.namelist()
    df_aq = pd.concat([pd.read_csv(zip_file.open(f)) for f in files if f.endswith('csv')])
    df_aq.to_csv("./PRSA2017_Data_20130301-20170228.csv", index=False)

df_aq['time'] = pd.to_datetime(df_aq.loc[:,['year','month','day','hour']])
df_aq = df_aq.rename(columns={'PM2.5':'PM2p5'})

df_aq_weekly = df_aq.\
    assign(date= lambda df: df['time'].astype('datetime64[D]') - pd.to_timedelta(df['time'].dt.dayofweek, unit='d')).\
    drop(columns= ['year','month','day','hour']).\
    groupby(['date','station']).\
    agg('mean').\
    reset_index().\
    sort_values(['station','date']).\
    reset_index()

# for training/validation split
SPLIT_DT = np.datetime64('2016-02-22') 

# get means/stds for preprocessing:
col_means = df_aq_weekly.loc[df_aq_weekly['date'] < SPLIT_DT,:].mean()
col_stds = df_aq_weekly.loc[df_aq_weekly['date'] < SPLIT_DT,:].std()
# -

df_aq_weekly.loc[:,['date','station','SO2','PM10','TEMP','PRES','DEWP']]

# + {"hidePrompt": true, "cell_type": "markdown"}
# #### Prepare our Dataset
#
# One of the key advantages of `torch-kalman` is the ability to train on a batch of time-serieses, instead of training a separate model for each individually. The `TimeSeriesDataset` is similar to PyTorch's native `TensorDataset`, with some useful metadata on the batch of time-serieses (the station names, the dates for each).

# +
# preprocess our measures of interest:
measures = ['SO2','PM10']
measures_pp = [m + '_log10_scaled' for m in measures]
df_aq_weekly[measures_pp] = np.log10(df_aq_weekly[measures] / col_means[measures])

# create a dataset:
dataset_all = TimeSeriesDataset.from_dataframe(
    dataframe=df_aq_weekly,
    dt_unit='W',
    measure_colnames=measures_pp,
    group_colname='station', 
    time_colname='date',
    pad_X=0.0
)

# Train/Val split:
dataset_train, dataset_val = dataset_all.train_val_split(dt=SPLIT_DT)
dataset_train, dataset_val
# -

# #### Specify our Model
#
# The `KalmanFilter` subclasses `torch.nn.Module`. We specify the model by passing `processes` that capture the behaviors of our `measures`.

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

# Here we're showing off a few useful features of `torch-kalman`:
#
# - We are training on a multivarite time-series: that is, our time-series has two measures (SO2 and PM10) and our model will capture correlations across these.
# - We are going to train on, and predictor for, multiple time-serieses (i.e. multiple stations) at once. 
# - We are allowing the amount of noise in the measure (i.e., the measure variance) to vary with the seasons, by passing 'seasonal' alias to `measure_var_predict`. (The `measure_var_predict` argument takes any `torch.nn.Module` that can be used for prediction, but 'seasonal' is an alias that tells the `KalmanFilter` to use a seasonal NN.)
#
# #### Train our Model
#
# When we call our KalmanFilter, we get predictions (a `StateBeliefOverTime`) which come with a mean and covariance, and so can be evaluated against the actual data using a (negative) log-probability critierion.

# +
kf_first.opt = LBFGS(kf_first.parameters(), lr=.20, max_eval=10)

def closure():
    kf_first.opt.zero_grad()
    pred = kf_first(
        dataset_train.tensors[0], 
        start_datetimes=dataset_train.start_datetimes, 
    )
    loss = -pred.log_prob(dataset_train.tensors[0]).mean()
    loss.backward()
    return loss

for epoch in range(15):
    train_loss = kf_first.opt.step(closure).item()
    with torch.no_grad():
        pred = kf_first(
            dataset_val.tensors[0], 
            start_datetimes=dataset_val.start_datetimes
        )
        val_loss = -pred.log_prob(dataset_val.tensors[0]).mean().item()
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss}, VAL LOSS {val_loss}")


# -

# #### Visualize the Results

# +
def inverse_transform(df: pd.DataFrame, col_means: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df['measure'] = df['measure'].str.replace('_log10_scaled','')
    std = (df['upper'] - df['lower']) / 1.96
    for col in ['mean','lower','upper','actual']:
        if col == 'mean':
            # bias correction:
            df[col] = df[col] + .5 * std ** 2
        df[col] = 10 ** df[col] # inverse log10
        df[col] *= df['measure'].map(col_means.to_dict()) # inverse scaling
    return df

pred = kf_first(
    dataset_train.tensors[0], 
    start_datetimes=dataset_train.start_datetimes,
    out_timesteps=dataset_all.tensors[0].shape[1]
)

df_pred = inverse_transform(pred.to_dataframe(dataset_all), col_means)

print(pred.plot(df_pred.query("group=='Changping'"), split_dt=SPLIT_DT))
# -

print(pred.plot(pred.to_dataframe(dataset_all, type='components').query("group=='Changping'"), split_dt=SPLIT_DT))

# #### Using Predictors
#
# Here, we'll use the weather to predict our measures of interest. We add these predictors by adding a `LinearModel` process to our model. `torch-kalman` also supports using *any* neural network to generate latent states for our model -- see the `NN` process.

# +
predictors = ['TEMP', 'PRES', 'DEWP']
predictors_pp = [x + '_scaled' for x in predictors]

df_aq_weekly[predictors_pp] = (df_aq_weekly[predictors] - col_means[predictors]) / col_stds[predictors]

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

# +
kf_pred = KalmanFilter(
    measures=measures_pp,
    processes=processes + [
        LinearModel(id=f'{m}_predictors', covariates=predictors_pp).add_measure(m) 
        for m in measures_pp
    ]
)

kf_pred.opt = LBFGS(kf_pred.parameters(), lr=.20, max_eval=10)

def closure():
    kf_pred.opt.zero_grad()
    y, X = dataset_train.tensors
    pred = kf_pred(y, predictors=X, start_datetimes=dataset_train.start_datetimes)
    loss = -pred.log_prob(y).mean()
    loss.backward()
    return loss

for epoch in range(20):
    train_loss = kf_pred.opt.step(closure).item()
    y, X = dataset_val.tensors
    with torch.no_grad():
        pred = kf_pred(y, predictors=X, start_datetimes=dataset_val.start_datetimes)
        val_loss = -pred.log_prob(y).mean().item()
    print(f"EPOCH {epoch}, TRAIN LOSS {train_loss}, VAL LOSS {val_loss}")

# +
y, _ = dataset_train.tensors # only input air-pollutant data from 'train' period
_, X = dataset_all.tensors # but provide exogenous predictors from both 'train' and 'validation' periods
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
print(pred.plot(df_components.query("(group=='Changping') & (process.str.endswith('predictors'))"), split_dt=SPLIT_DT))


