# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# +
import torch
import copy

from torchcast.kalman_filter import KalmanFilter
from torchcast.utils.data import TimeSeriesDataset, complete_times

import numpy as np
import pandas as pd

np.random.seed(2021-1-21)
torch.manual_seed(2021-1-21)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
# -

GROUPS = [f'MT_00{i}' for i in range(1,7) if i!=3]
GROUPS

# ## Using NN's for Long-Range Forecasts: Electricity Data
#
# Here's we'll show how to manage a few related obstacles that can arise in forecasting multiple series:
#
# - The time-serieses are diverse -- if we want a train a single model on all of them, we'll need to give it the capacity to capture their differences. Here we'll show how to 
# - The time-serieses are long. Running the model through thousands of timesteps on every forward/backward pass is computationally expensive relatie to what it gains us.

# +
import requests
from zipfile import ZipFile
from io import BytesIO

# response =\
#     requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip')

# with ZipFile(BytesIO(response.content)) as f:
with ZipFile("./LD2011_2014.txt.zip") as f:
    df_raw = pd.read_table(f.open('LD2011_2014.txt'), sep=";", decimal=",")

# +
df_elec = df_raw.loc[:,['Unnamed: 0'] + GROUPS].\
    melt(id_vars=['Unnamed: 0'], value_name='kW', var_name='group').\
    assign(time = lambda df_elec: df_elec['Unnamed: 0'].astype('datetime64[h]')).\
    groupby(['group','time'])\
    ['kW'].mean().\
    reset_index()

df_elec = df_elec.\
    query("kW > 0").\
    pipe(complete_times, group_colname='group')

df_elec['kW_sqrt'] = np.sqrt(df_elec['kW'])
df_elec
# -

SPLIT_DT = np.datetime64('2014-01-01')

# ### Standard Approach

# +
from torchcast.utils.features import add_season_features
    
df_elec_train = df_elec.\
    query("time <= @SPLIT_DT").\
    assign(
        ym = lambda df: df['time'].dt.year.astype('str') + df['time'].dt.month.astype('str').str.zfill(2),
        group_ym = lambda df: df['group'] + ':' + df['ym']
    )


# add season features:
df_elec_train = df_elec_train.\
    pipe(add_season_features, K=3, period='weekly').\
    pipe(add_season_features, K=6, period='yearly').\
    pipe(add_season_features, K=5, period='daily')

season_cols =\
    df_elec_train.columns[df_elec_train.columns.str.endswith('_sin')|df_elec_train.columns.str.endswith('_cos')]

df_elec_train
# -

dataset_train = TimeSeriesDataset.from_dataframe(
    dataframe=df_elec_train,
    dt_unit='h',
    y_colnames=['kW_sqrt'],
    X_colnames=season_cols,
    group_colname='group_ym', 
    time_colname='time',
    pad_X=True,
    device=DEVICE
)
assert not torch.isnan(dataset_train.tensors[1]).any()
dataset_train

# establish a mapping from group-names to integers for torch.nn.Embedding:
group_id_mapping = {gn : i for i, gn in enumerate(np.unique(df_elec_train['group']))}

# +
from torchcast.process import LocalTrend, LocalLevel, Season, LinearModel
from torchcast.covariance import Covariance

processes = [
        LocalTrend(id='trend'),
        LocalLevel(id='level', decay=True),
        Season(id='hour_in_day', period=24, dt_unit='h', K=6, decay=True),
        LinearModel(id='lm', predictors=dataset_train.measures[1])
    ]
mvar_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=1,
    padding_idx=0
)
pvar_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=Covariance.for_processes(processes).param_rank,
    padding_idx=0
)
# -

kf_lm = KalmanFilter(
    measures=dataset_train.measures[0], 
    processes=processes,
    measure_covariance=Covariance.for_measures(dataset_train.measures[0], predict_variance=mvar_nn),
    process_covariance=Covariance.for_processes(processes, predict_variance=pvar_nn)
)
kf_lm

# +
# create a network that takes group ids and produces the initial prediction
initial_state_nn = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=len(kf_lm.initial_mean),
    padding_idx=0
)
# use gentler inits than the default:
with torch.no_grad():
    initial_state_nn.weight[:] *= .01

# register these parameters in our KF so that they show up in `kf_lm.parameters()` (and therefore the optimizer)
for k, v in initial_state_nn.named_parameters():
    kf_lm.register_parameter(f'initial_state_nn__{k}',v)
    
#
kf_lm.to(DEVICE)


# +
def get_group_ids(dataset):
    group_names = pd.Series(dataset.group_names).str.split(":", expand=True)[0]
    return torch.as_tensor([group_id_mapping[gn] for gn in group_names], device=DEVICE)

def _get_initial_state():
    return initial_state_nn(get_group_ids(dataset_train)) + kf_lm.initial_mean[None,:]

y, X = dataset_train.tensors
kf_lm.fit(
    y, 
    X=X, 
    measure_covariance__X=get_group_ids(dataset_train),
    process_covariance__X=get_group_ids(dataset_train),
    n_step=int(24*7.5),
    every_step=False,
    callable_kwargs={'initial_state' : _get_initial_state},
    start_offsets=dataset_train.start_datetimes
)

# +
dataset_full = TimeSeriesDataset.from_dataframe(
    dataframe=df_elec.\
        pipe(add_season_features, K=3, period='weekly').\
        pipe(add_season_features, K=6, period='yearly').\
        pipe(add_season_features, K=5, period='daily'),
    dt_unit='h',
    y_colnames=['kW_sqrt'],
    X_colnames=season_cols,
    group_colname='group',
    time_colname='time'
)
assert not torch.isnan(dataset_full.tensors[1]).any()
dataset_full

with torch.no_grad():
    y, _ = dataset_full.train_val_split(dt=SPLIT_DT)[0].tensors
    _, X = dataset_full.tensors
    pred_lm = kf_lm(
            y,
            X=X,
            measure_covariance__X=get_group_ids(dataset_full),
            process_covariance__X=get_group_ids(dataset_full),
            initial_state=initial_state_nn(get_group_ids(dataset_full)),
            out_timesteps=X.shape[1],
            start_offsets=dataset_full.start_datetimes
        )
# -

print(pred_lm.plot(pred_lm.to_dataframe(dataset_full).query("group=='MT_005' & time>=@SPLIT_DT"), split_dt=SPLIT_DT))


print(
    pred_lm.plot(pred_lm.to_dataframe(dataset_full, type='components').query("group=='MT_005'"))
)

# +
# TODO:

print(
    pred_lm.plot(pred_lm.to_dataframe(dataset_full).\
                 query('group=="MT_001" & time.between("2012-07-01","2012-08-01")')) +
    geom_line(aes(y='actual', color='time.dt.day_name()'))
)
# -

# ### Using a Neural Network

# +
from torchcast.process import NN

processes = [
    LocalTrend(id='trend'),
    LocalLevel(id='level', decay=True),
    Season(id='hour_in_day', period=24, dt_unit='h', K=6, decay=True),
    NN(
        id='nn', 
        nn=torch.nn.Sequential(
            torch.nn.Linear(len(dataset_train.measures[1]), 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20,15, bias=False)
        )
    )
]

mvar_nn2 = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=1,
    padding_idx=0
)
pvar_nn2 = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=Covariance.for_processes(processes).param_rank,
    padding_idx=0
)

kf_nn = KalmanFilter(
    measures=dataset_train.measures[0], 
    processes=processes,
    measure_covariance=Covariance.for_measures(dataset_train.measures[0], predict_variance=mvar_nn2),
    process_covariance=Covariance.for_processes(processes, predict_variance=pvar_nn2)
)

initial_state_nn2 = torch.nn.Embedding(
    len(group_id_mapping),
    embedding_dim=len(kf_nn.initial_mean),
    padding_idx=0
)
with torch.no_grad():
    initial_state_nn2.weight[:] *= .01

for k, v in initial_state_nn2.named_parameters():
    kf_nn.register_parameter(f'initial_state_nn__{k}',v)


# +
def _get_initial_state2():
    return initial_state_nn2(get_group_ids(dataset_train)) + kf_nn.initial_mean[None,:]

kf_nn.loss_history = []
y, X = dataset_train.tensors
kf_nn.fit(
    y, 
    X=X, 
    measure_covariance__X=get_group_ids(dataset_train),
    process_covariance__X=get_group_ids(dataset_train),
    n_step=int(24*7.5),
    every_step=False,
    callbacks=[lambda l: kf_nn.loss_history.append(l)],
    callable_kwargs={'initial_state' : _get_initial_state2},
    start_offsets=dataset_train.start_datetimes
)
