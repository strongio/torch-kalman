# ---
# jupyter:
#   jupytext:
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

# + nbsphinx="hidden"
# # !pip install git+https://github.com/strongio/torch-kalman.git@feature/rename#egg=torchcast
# # !pip install torch_optimizer

import torch
import copy

import matplotlib.pyplot as plt

from torchcast.kalman_filter import KalmanFilter
from torchcast.utils.data import (
    TimeSeriesDataset, TimeSeriesDataLoader, complete_times, nanmean
)

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

BASE_DIR = './'#drive/MyDrive'

import os
if 'drive/MyDrive' in BASE_DIR and not os.path.exists(BASE_DIR):
    from google.colab import drive
    drive.mount('/content/drive')
# -

# # Using NN's for Long-Range Forecasts: Electricity Data
#
# Here's we'll show how to manage some obstacles that can arise in forecasting real data:
#
# - The time-serieses are complex. We can't treat individual components of the series as independent. For example, there is no 'hour-in-day' component that's independent of the 'day-of-week' or 'day-in-year' component -- everything is interrelated. Here we'll show how to do this by leveraging `torchcast`'s ability to integrate with any PyTorch neural-network.
# - The time-serieses are long. Running the model through thousands of timesteps on every forward/backward pass is computationally expensive relative to what it gains us. We can avoid this by splitting each series into sub-series.
#
# We'll use a dataset from the [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), which consists of electricity-usage for 370 locations, taken every 15 minutes (we'll downsample to hourly).

# + nbsphinx="hidden"
try:
    df_elec = pd.read_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), parse_dates=['time'])
except FileNotFoundError:
    import requests
    from zipfile import ZipFile
    from io import BytesIO
    response =\
        requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip')

    with ZipFile(BytesIO(response.content)) as f:
        df_raw = pd.read_table(f.open('LD2011_2014.txt'), sep=";", decimal=",")

    # melt, collect to hourly:
    df_elec = df_raw.\
        melt(id_vars=['Unnamed: 0'], value_name='kW', var_name='group').\
        assign(time = lambda df_elec: df_elec['Unnamed: 0'].astype('datetime64[h]')).\
        groupby(['group','time'])\
        ['kW'].mean().\
        reset_index()

    df_elec.\
        loc[df_elec['time']>=df_elec['group'].map(df_elec.query("kW>0").groupby('group')['time'].min().to_dict()),:].\
        reset_index(drop=True).\
        to_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), index=False)
        
    df_elec = pd.read_csv(os.path.join(BASE_DIR, "df_electricity.csv.gz"), parse_dates=['time'])

SUBSET = 50
np.random.seed(2021-1-21)
torch.manual_seed(2021-1-21)
# -

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# ## Data-Cleaning
#
# Our dataset consists of hourly kW readings for multiple buildings:

df_elec.head()

# Electricity-demand data can be challenging because of its complexity. In traditional forecasting applications, we divide our model into siloed processes that each contribute to separate 'behaviors' of the time-series. For example:
#
# - Hour-in-day effects
# - Day-in-week effects
# - Season-in-year effects
# - Weather effects
#
# However, with electricity data, it's limiting to model these separately, because **these effects all interact**: the impact of hour-in-day depends on the day-of-week, the impact of the day-of-week depends on the season of the year, etc.
#
# We can plot some examples to get an initial glance at this complexity.

df_elec.query("group=='MT_001'").plot('time','kW',figsize=(20,5))

# Some groups have data that isn't really appropriate for modeling -- for example, exhibiting near-zero variation:

df_elec.query("group=='MT_003'").plot('time','kW',figsize=(20,5))

# For some rudimentary cleaning, we'll remove these kinds of regions of 'flatness':

# +
# calculate rolling std-dev:
df_elec['roll_std'] = 0
for g, df in tqdm(df_elec.groupby('group')):
    df_elec.loc[df.index, 'roll_std'] = df['kW'].rolling(48).std()
df_elec.loc[df_elec.pop('roll_std') < .25, 'kW'] = float('nan')

group_missingness = df_elec.assign(missing=lambda df: df['kW'].isnull()).groupby('group')['missing'].mean()

df_elec = df_elec.loc[df_elec['group'].map(group_missingness) < .01,:].reset_index(drop=True)
# -

# For simplicity we'll just drop buildings that are flat in this way for a non-trivial amount of time.
#
# We'll also subset to groups with at least 2 years of data, so we're guaranteed enough data for forecasting:

# +
df_group_summary = df_elec.\
  groupby('group')\
    ['time'].agg(['min','max']).\
    reset_index().\
    assign(history_len = lambda df: (df['max'] - df['min']).dt.days)

train_groups = sorted(df_group_summary.query("history_len >= 730")['group'])

#
train_groups = train_groups[:SUBSET or -1]
df_elec = df_elec.loc[df_elec['group'].isin(train_groups),:].reset_index(drop=True)
# -

# ### Training/Validation/Test
#
# We'll split the data at 2014. For half the groups, this will be used as validation data; for the other half, it will be used as test data.

SPLIT_DT = np.datetime64('2014-01-01')
df_elec['_use_2014_as_test'] = (df_elec['group'].str.replace('MT_','').astype('int') % 2) == 0
df_elec['dataset'] = 'train'
df_elec.loc[(df_elec['time'] >= SPLIT_DT) & df_elec['_use_2014_as_test'],'dataset'] = 'test'
df_elec.loc[(df_elec['time'] >= SPLIT_DT) & ~df_elec.pop('_use_2014_as_test'),'dataset'] = 'val'
#df_elec['dataset'].value_counts()

# ## A Standard Forecasting Approach
#
# First, let's try a forecasting approach that's fairly 'standard':
#
# - We'll build a model for a single series (instead of sharing a model across multiple series)
# - Each 'process' (e.g. day-in-week, day-in-year, hour-in-day) we'll be assumed to have a separate, independent influence on the data.

# +
from torchcast.process import LocalLevel, LocalTrend, Season

kf = KalmanFilter(
    measures=['kW_sqrt'], 
    processes=[
        # seasonal processes:
        Season(id='day_in_week', period=24*7, dt_unit='h', K=3, process_variance=False),
        Season(id='day_in_year', period=24*365.25, dt_unit='h', K=8, process_variance=False),
        Season(id='hour_in_day', period=24, dt_unit='h', K=8, process_variance=False),
        # long-running trend:
        LocalTrend(id='trend'),
        # 'local' processes: allow temporary deviations that decay 
        LocalLevel(id='level', decay=True),
        Season(id='hour_in_day2', period=24, dt_unit='h', K=6, decay=True),
    ]
)

# +
# build our dataset

df_elec['kW_sqrt'] = np.sqrt(df_elec['kW'])

# for efficiency of training, we split this single group into multiple groups
# this is helpful since pytorch has a non-trivial overhead for separate tensors --
# i.e., it scales well with an increasing batch-size (fewer, but bigger, tensors), 
# but poorly with an increasing time-seriees length (smaller, but more, tensors)
df_elec['gym'] =\
     df_elec['group'] + ":" +\
     df_elec['time'].dt.year.astype('str') + "_" +\
     df_elec['time'].dt.month.astype('str').str.zfill(2)

train_MT_052 = TimeSeriesDataset.from_dataframe(
    df_elec.query("group == 'MT_052'").query("dataset == 'train'"),
    group_colname='gym', 
    time_colname='time',
    measure_colnames=['kW_sqrt'],
    dt_unit='h'
)
train_MT_052 = train_MT_052.to(DEVICE)
train_MT_052

# +
kf.to(DEVICE)

try:
    kf.load_state_dict(torch.load(os.path.join(BASE_DIR, "kf_standard.pt"), map_location=DEVICE))
except FileNotFoundError:
    kf.fit(
        train_MT_052.tensors[0],
        start_offsets=train_MT_052.start_datetimes,
        n_step=int(24 * 7.5),
        every_step=False
    )
    torch.save(kf.state_dict(), os.path.join(BASE_DIR, "kf_standard.pt"))
# -

# ### Model-Evaluation
#
# How does this standard model perform? Plotting the forecasts vs. actual suggests some issues:

eval_MT_052 = TimeSeriesDataset.from_dataframe(
    df_elec.query("group == 'MT_052'"),
    group_colname='group', 
    time_colname='time',
    measure_colnames=['kW_sqrt'],
    dt_unit='h'
).to(DEVICE)
with torch.no_grad():
    y = eval_MT_052.train_val_split(dt=SPLIT_DT)[0].tensors[0]
    pred = kf(
            y, 
            start_offsets=eval_MT_052.start_datetimes,
            out_timesteps=y.shape[1] + 24*365.25,
        ) 
    df_pred52 = pred.to_dataframe(eval_MT_052)
df_pred52 = df_pred52.loc[~df_pred52['actual'].isnull(),:].reset_index(drop=True)
pred.plot(df_pred52, split_dt=SPLIT_DT)

# With hourly data, visualizing long-range forecasts in this way isn't very illuminating: it's just really hard to see the data! Let's try splitting it into weekdays vs. weekends and daytimes vs. nightimes:

# +
df_pred52_split = df_pred52.\
           query("(time.dt.hour==8) | (time.dt.hour==20)").\
           assign(weekend = lambda df: df['time'].dt.weekday.isin([5,6]).astype('int'),
                  night = lambda df: (df['time'].dt.hour == 8).astype('int')).\
        reset_index(drop=True)

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
for (weekend, night), df in df_pred52_split.groupby(['weekend','night']):
    df.plot('time','actual', ax=axes[weekend,night], linewidth=.5, color='black')
    df.plot('time', 'mean', ax=axes[weekend,night], alpha=.75, color='red')
    axes[weekend,night].axvline(x=SPLIT_DT, color='black', ls='dashed')
    axes[weekend,night].set_title("{}, {}".format('Weekend' if weekend else 'Weekday', 'Night' if night else 'Day'))
plt.tight_layout()
# -

# This helps us understand the problem: the annual seasonal pattern is very different from daytimes and nighttimes, but the model isn't (and can't be) capturing that. For example, it incorrectly forecasts a 'hump' during summer days and weekend nights, even though this hump is really only present on weekday nights. The model only allows for a single seasonal pattern, rather than a separate one for different times of the day and days of the week.

# ## Incorporating a Neural Network
#
# If the problem is that our time-series model isn't allowing for interactions among the components of our time-series model, then a natural approach to allowing for these interactions is a neural network.
#
# To make this work, we likely don't want build a model of a single series. Instead, we want to learn across multiple series, so that our network can build representations of patterns that are shared across multiple buildings.

# +
# prepare our dataset: add dummy features that capture the annual, weekly, and daily seasonal patterns:
from torchcast.utils import add_season_features

df_elec = df_elec.\
    pipe(add_season_features, K=3, period='weekly').\
    pipe(add_season_features, K=8, period='yearly').\
    pipe(add_season_features, K=8, period='daily')

season_cols =\
    df_elec.columns[df_elec.columns.str.endswith('_sin')|df_elec.columns.str.endswith('_cos')]

df_elec['kW_sqrt'] = np.sqrt(df_elec['kW'])

dataset_kwargs = dict(
    dt_unit='h',
    y_colnames=['kW_sqrt'],
    X_colnames=season_cols,
    time_colname='time'
)

trainval_batches = TimeSeriesDataLoader.from_dataframe(
    df_elec.query("dataset != 'test'"), 
    group_colname='group', 
    **dataset_kwargs,
    batch_size=20
)

val_batch = TimeSeriesDataset.from_dataframe(
    df_elec.query("dataset == 'val'"), 
    group_colname='group', 
    **dataset_kwargs
)
# -

# ### Neural-Network for Time-Series
#
# Since `torchcast` is built on top of PyTorch, we can train a model end-to-end: i.e. we could start with a neural network with random-inits, plug it into a `KalmanFilter`, and train the whole thing.
#
# In practice, we'll generally want to take a neural-network that has already been partially or fully trained to generate sensible predictions. A standard feedforward network will take many many more epochs to train than a `KalmanFilter`, so it's helpful to keep the two models separate intially.
#
# Here we'll use a network that combines a per-building embedding with a standard architecture that's shared across buildings. This allows the network's shared layers to find representations that are shared across many buildings.
#
# Note the network here uses only calendar features (fourier-transforms on daily/weekly/yearly seasonal periods) to generate predictions. This could be expanded depending on the available data: e.g. holiday-indicators, weather-data, etc.

# +
# mapping from group-names to integers for torch.nn.Embedding:
group_id_mapping = {gn : i for i, gn in enumerate(np.unique(df_elec['group']))}

def get_group_ids(dataset):
    group_names = pd.Series(dataset.group_names).str.split(":", expand=True)[0]
    return torch.as_tensor([group_id_mapping[gn] for gn in group_names], device=DEVICE)

# scaling
group_means = df_elec.query("time < @SPLIT_DT").groupby('group')['kW_sqrt'].mean().to_dict()
group_stds = df_elec.query("time < @SPLIT_DT").groupby('group')['kW_sqrt'].mean().to_dict()
def standardize_by_group(tensor, dataset):
    group_names = pd.Series(dataset.group_names).str.split(":", expand=True)[0]
    means = torch.as_tensor([group_means[gn] for gn in group_names], device=DEVICE)
    stds = torch.as_tensor([group_stds[gn] for gn in group_names], device=DEVICE)
    return (tensor - means[:,None,None]) / stds[:,None,None]


# +
# helper class for combining shared layers with per-building embedding:
class PerGroupNN(torch.nn.Module):
    def __init__(self, shared: torch.nn.Module, embedding: torch.nn.Module):
        super().__init__()
        self.shared = shared
        self.embedding = embedding

    def forward(self, X, group_ids):
        shared_weights = self.shared(X)
        per_group = self.embedding(group_ids)
        per_group_w, per_group_b = torch.split(
            per_group.unsqueeze(1), 
            [self.embedding.embedding_dim - 1, 1], 
            -1
        )
        outw = shared_weights.unsqueeze(-2) @ per_group_w.unsqueeze(-1)
        return outw.squeeze(-1) + per_group_b 

per_group_nn = PerGroupNN(
    shared = torch.nn.Sequential(
        torch.nn.Linear(len(season_cols), 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 20)
    ),
    embedding = torch.nn.Embedding(
        num_embeddings=len(group_id_mapping),
        embedding_dim=21
    )
)
per_group_nn.to(DEVICE)

try:
    per_group_nn.load_state_dict(torch.load(os.path.join(BASE_DIR, "per_group_nn.pt"), map_location=DEVICE))
except FileNotFoundError:
    from IPython import display
    per_group_nn.optimizer = torch.optim.Adam(per_group_nn.parameters())

    per_group_nn.loss_history = []
    per_group_nn.val_history = []
    for epoch in range(1500):
        epoch_loss = 0
        for batch in trainval_batches:
            batch, _ = batch.train_val_split(dt=SPLIT_DT)
            y, X = batch.to(DEVICE).tensors
            y = standardize_by_group(y, batch)
            nan_mask = torch.isnan(y)
            group_ids = get_group_ids(batch)
            try:
                pred = per_group_nn(X, group_ids=group_ids)
                loss = torch.mean((pred[~nan_mask] - y[~nan_mask]) ** 2)
                loss.backward()
                per_group_nn.optimizer.step()
                epoch_loss += loss.item()
            finally:
                per_group_nn.optimizer.zero_grad(set_to_none=True)
        per_group_nn.loss_history.append(epoch_loss / len(train_batches))

        with torch.no_grad():
            y, X = val_batch.to(DEVICE).tensors
            y = standardize_by_group(y, val_batch)
            nan_mask = torch.isnan(y)
            group_ids = get_group_ids(val_batch)
            pred = per_group_nn(X, group_ids=group_ids)
            val_loss = torch.mean((pred[~nan_mask] - y[~nan_mask]) ** 2)
            per_group_nn.val_history.append(val_loss.item())

        if epoch > 10:
            plt.close()
            display.clear_output(wait=True)
            fig, axes = plt.subplots(ncols=2, figsize=(10,5))
            pd.Series(per_group_nn.loss_history[10:]).plot(ax=axes[0], logy=True)
            pd.Series(per_group_nn.val_history[10:]).plot(ax=axes[1], logy=True)
            display.display(plt.gcf())
# -

# ### Training our Hybrid Forecasting Model
#
# Now that we have a network that

# +
from torchcast.process import NN, LocalLevel, LocalTrend, Season
from torchcast.covariance import Covariance

shared_nn = copy.deepcopy(per_group_nn.shared)

processes = [
    LocalTrend(id='trend'),
    LocalLevel(id='level', decay=True),
    Season(id='hour_in_day', period=24, dt_unit='h', K=6, decay=True),
    NN(id='nn', nn=shared_nn)
]

mvar_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=1,
)
pvar_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=Covariance.for_processes(processes).param_rank,
)

kf_nn = KalmanFilter(
    measures=['kW_sqrt'], 
    processes=processes,
    measure_covariance=Covariance.for_measures(['kW_sqrt'], predict_variance=mvar_nn),
    process_covariance=Covariance.for_processes(processes, predict_variance=pvar_nn)
)

kf_nn.initial_state_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=len(kf_nn.initial_mean),
)
with torch.no_grad():
    kf_nn.initial_state_nn.weight[:] *= .01
kf_nn.initial_mean = None

kf_nn.to(DEVICE)
# -

try:
    kf_nn.load_state_dict(torch.load(os.path.join(BASE_DIR,"kf_nn.pt"), map_location=DEVICE))
except FileNotFoundError:
    from IPython import display
    train_batches = TimeSeriesDataLoader.from_dataframe(
        df_elec.query("dataset == 'train'"), 
        group_colname='subgroup', 
        **dataset_kwargs, 
        batch_size=100,
        shuffle=True
    )
    len(train_batches)

    from torch_optimizer import Adahessian
    kf_nn.optimizer = Adahessian([
        {'params' : [p for n,p in kf_nn.named_parameters() if 'processes.nn.'     in n], 'lr' : .001},
        {'params' : [p for n,p in kf_nn.named_parameters() if 'processes.nn.' not in n], 'lr' : .50}
    ])

    kf_nn.loss_history = []
    kf_nn.val_history = []
    for epoch in range(250):
        # training:
        train_loss = 0
        for batch in tqdm(train_batches, total=len(train_batches), desc=f"Epoch {epoch}"):
            y, X = batch.to(DEVICE).tensors
            group_ids = get_group_ids(batch)
            try:
                pred = kf_nn(
                    y, 
                    X=X,
                    measure_covariance__X=group_ids,
                    process_covariance__X=group_ids,
                    initial_state=kf_nn.initial_state_nn(group_ids),
                    start_offsets=batch.start_datetimes,
                    n_step=int(24 * 7.5),
                    every_step=False
                )
                loss = -pred.log_prob(y).mean()
                loss.backward(create_graph=True)
                kf_nn.optimizer.step()
                train_loss += loss.item()
            finally:
                kf_nn.optimizer.zero_grad(set_to_none=True)
        train_loss /= len(train_batches)
        kf_nn.loss_history.append(train_loss)

        # validation:
        with torch.no_grad():
            y, X = val_batch.to(DEVICE).tensors
            group_ids = get_group_ids(val_batch)
            pred = kf_nn(
                    y, 
                    X=X,
                    measure_covariance__X=group_ids,
                    process_covariance__X=group_ids,
                    initial_state=kf_nn.initial_state_nn(group_ids),
                    start_offsets=val_batch.start_datetimes,
                    n_step=int(24 * 7.5),
                    every_step=False
                )
            val_errs = nanmean((pred.means - y).abs(), dim=1) / nanmean(y, dim=1)
            kf_nn.val_history.append(val_errs.mean().item())

        display.clear_output(wait=True)
        if epoch > 2:
            plt.close()
            fig, axes = plt.subplots(ncols=2,figsize=(10,5))
            pd.Series(kf_nn.loss_history[2:]).plot(ax=axes[0])
            pd.Series(kf_nn.val_history[2:]).plot(ax=axes[1])
            display.display(plt.gcf())
        torch.save(kf_nn.state_dict(), os.path.join(BASE_DIR,"kf_nn.pt"))

withtest_batches = TimeSeriesDataLoader.from_dataframe(
    df_elec,
    group_colname='group', 
    **dataset_kwargs,
    batch_size=20
)
df_pred_nn = []
with torch.no_grad():
    for eval_batch in tqdm(withtest_batches):
        eval_batch = eval_batch.to(DEVICE)
        y = eval_batch.train_val_split(dt=SPLIT_DT)[0].tensors[0]
        X = eval_batch.tensors[1]
        group_ids = get_group_ids(eval_batch)
        pred_nn = kf_nn(
                y, 
                X=X,
                measure_covariance__X=group_ids,
                process_covariance__X=group_ids,
                initial_state=kf_nn.initial_state_nn(group_ids),
                start_offsets=eval_batch.start_datetimes,
                out_timesteps=X.shape[1],
            ) 
        df_pred_nn.append(pred_nn.to_dataframe(eval_batch))
df_pred_nn = pd.concat(df_pred_nn)
df_pred_nn = df_pred_nn.loc[~df_pred_nn['actual'].isnull(),:].reset_index(drop=True)

# +
df_pred52_nn = df_pred_nn.\
           query("group=='MT_052'").\
           query("(time.dt.hour==8) | (time.dt.hour==20)").\
           assign(weekend = lambda df: df['time'].dt.weekday.isin([5,6]).astype('int'),
                  night = lambda df: (df['time'].dt.hour == 8).astype('int')).\
        reset_index(drop=True)

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
for (weekend, night), df in df_pred52_nn.groupby(['weekend','night']):
    df.plot('time','actual', ax=axes[weekend,night], linewidth=.5, color='black')
    df.plot('time', 'mean', ax=axes[weekend,night], alpha=.75, color='red')
    axes[weekend,night].axvline(x=SPLIT_DT, color='black', ls='dashed')
    axes[weekend,night].set_title("{}, {}".format('Weekend' if weekend else 'Weekday', 'Night' if night else 'Day'))
plt.tight_layout()

# + nbsphinx="hidden"
"""
- MT_029 -- need (way) higher K for annual season?
- MT_018, MT_024 -- still systematic bias in certain parts of the day. why?
""";
