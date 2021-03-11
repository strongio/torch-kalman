# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3.7.9 64-bit ('torch-kalman')
#     metadata:
#       interpreter:
#         hash: e593fb664dc87c5fcb156f9082aabc58b575809650fd778da08a33de5f780190
#     name: python3
# ---

# +
from sklearn.preprocessing import StandardScaler
import datetime
import os
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from tqdm.auto import tqdm

import torch

from torch_kalman.covariance import Covariance
from torch_kalman.kalman_filter.base import KalmanFilter
from torch_kalman.process import LocalTrend, LocalLevel, FourierSeason, LinearModel
from torch_kalman.utils.data import TimeSeriesDataset

from pybaseball import schedule_and_record

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sb
sb.set(style='white', color_codes=True)

blue, red, yellow, green, orange, pink, purple, brown = sb.xkcd_palette(
    ['windows blue', 'pale red', 'amber', 'deep green', 'reddish orange', 'pink', 'purple', 'brown']
)
# -

DATA_DIR = '.'


# +
def load_riders_data(start_date: str = None, end_date: str = None):
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    assert isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp)
    riders_df = pd.read_csv(os.path.join(DATA_DIR, 'ridership.csv'))
    riders_df['date'] = pd.to_datetime(riders_df['date'])
    riders_df = riders_df.drop_duplicates(['date', 'stationname'])
    riders_df = riders_df.sort_values(['date', 'station_id']).reset_index(drop=True)

    if start_date is not None:
        riders_df = riders_df[riders_df['date'] >= start_date]
    if end_date is not None:
        riders_df = riders_df[riders_df['date'] <= end_date]

    station_info = pd.read_csv(os.path.join(DATA_DIR, 'station_info.csv'))
    station_info.columns = station_info.columns.str.lower()
    line_map = {'g': 'green', 'brn': 'brown', 'p': 'purple', 'pexp': 'purple_express', 'y': 'yellow', 'pnk': 'pink', 'o': 'orange'}
    station_info = station_info.rename(line_map, axis=1)

    station_id_cols = ['map_id', 'station_name', 'station_descriptive_name']
    lines = ['red', 'blue', 'green', 'brown', 'purple', 'purple_express', 'yellow', 'pink', 'orange']
    station_lines = pd.DataFrame(columns=station_id_cols + lines + ['n_lines'])
    for station_id, dat in station_info.groupby('map_id'):
        station_dat = dat[station_id_cols].iloc[0].values.tolist()
        line_dat = [any(dat[line]) for line in lines]
        row = station_dat + line_dat + [sum(line_dat)]
        station_lines = pd.concat([station_lines, pd.DataFrame([row], columns=station_id_cols + lines + ['n_lines'])])

    riders_df = riders_df.merge(station_lines, left_on='station_id', right_on='map_id')
    riders_df.loc[riders_df['daytype'] != 'W', 'purple_express'] = False

    return riders_df


def process_riders_data(riders_df):
    lines = ['red', 'blue', 'green', 'brown', 'purple', 'purple_express', 'yellow', 'pink', 'orange']
    # For now, just count rides at transfer (i.e. multi-line) stations for each line - need to account for this later though
    df = pd.DataFrame(columns=['date', 'rides', 'line'])
    for line in lines:
        line_df = riders_df[riders_df[line]]
        line_df['normed_rides'] = line_df['rides'] / line_df['n_lines']
        daily_line_data = line_df.groupby('date')['normed_rides'].sum().to_frame().reset_index()
        daily_line_data = daily_line_data.rename({'normed_rides': 'rides'}, axis=1)
        daily_line_data['rides'] = daily_line_data['rides'].astype(np.int64)
        daily_line_data['line'] = line
        df = pd.concat([df, daily_line_data])

    return df


def load_weather_data():
    weather_data = pd.read_csv('../data/cta/weather_data.csv')
    weather_data.columns = weather_data.columns.str.lower()
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    return weather_data


def add_date_flags(daily_df: pd.DataFrame):
    assert 'date' in daily_df.columns
    daily_df['is_weekday'] = (daily_df['date'].dt.weekday < 5).astype(np.int64)
    daily_df['is_weekend'] = (daily_df['date'].dt.weekday >= 5).astype(np.int64)

    holidays = USFederalHolidayCalendar().holidays()
    daily_df['is_holiday'] = df['date'].isin(holidays).astype(np.int64)

    cubs_home_games = pd.Series()
    for year in daily_df['date'].dt.year.unique():
        sched = schedule_and_record(year, 'CHC')
        home_dates = sched[sched['Home_Away'] == 'Home']['Date'].apply(lambda r: pd.Timestamp(' '.join([r.split('(')[0].split(',')[1], str(year)])))
        home_dates = home_dates.drop_duplicates()
        cubs_home_games = cubs_home_games.append(home_dates, ignore_index=True)

    daily_df['is_cubs_game'] = df['date'].isin(cubs_home_games).astype(np.int64)

    sox_home_games = pd.Series()
    for year in daily_df['date'].dt.year.unique():
        sched = schedule_and_record(year, 'CHW')
        home_dates = sched[sched['Home_Away'] == 'Home']['Date'].apply(lambda r: pd.Timestamp(' '.join([r.split('(')[0].split(',')[1], str(year)])))
        home_dates = home_dates.drop_duplicates()
        sox_home_games = sox_home_games.append(home_dates, ignore_index=True)

    daily_df['is_sox_game'] = df['date'].isin(sox_home_games).astype(np.int64)
    return daily_df


# +
DT_UNIT = 'D'
START_DT = pd.Timestamp('2008-06-01')
END_DT = pd.Timestamp('2018-12-31')
SPLIT_DT = pd.Timestamp('2017-06-01')

color_map = {
    'red': red,
    'blue': blue,
    'green': green,
    'brown': brown,
    'pink': pink,
    'purple': purple,
    'orange': orange,
    'yellow': yellow
}

# +
riders_df = load_riders_data(start_date=START_DT, end_date=END_DT)
df = process_riders_data(riders_df)
df = add_date_flags(df)

weather_data = load_weather_data()
df = df.merge(weather_data, on='date')

# +

# Filter to a few lines
lines = ('red', 'blue', 'green', 'brown', 'pink', 'orange', 'yellow', 'purple', 'purple_express')
df = df[df['line'].isin(lines)]

measures = ['rides']
predictors = ['tmax', 'prcp', 'is_weekday', 'is_weekend', 'is_holiday', 'is_cubs_game']
normed_measures = []
normed_predictors = []
scalers = {}
for measure in measures + predictors:
    new_col = f'{measure}_scaled'
    normed_measures.append(new_col) if measure in measures else normed_predictors.append(new_col)

    fit_dat = df[df['date'] <= SPLIT_DT][measure].values.reshape(-1, 1)
    transform_dat = df[measure].values.reshape(-1, 1)

    scaler = StandardScaler().fit(fit_dat)
    df[new_col] = scaler.transform(transform_dat).flatten()
    scalers[measure] = scaler

dataset_all = TimeSeriesDataset.from_dataframe(
    dataframe=df,
    dt_unit=DT_UNIT,
    y_colnames=normed_measures,
    X_colnames=normed_predictors,
    group_colname='line',
    time_colname='date'
)
# -

dataset_train, dataset_val = dataset_all.train_val_split(dt=SPLIT_DT.to_numpy())

# +
processes = []
for measure in normed_measures:
    processes.extend([
        LocalTrend(id=f'{measure}_trend', measure=measure),
        LocalLevel(id=f'{measure}_local_level', decay=(.99, 1.00), measure=measure),
        FourierSeason(
            id=f'{measure}_day_of_week',
            period=7.,
            dt_unit=DT_UNIT,
            K=7,
            measure=measure
        ),
        FourierSeason(
            id=f'{measure}_day_of_year',
            period=365.,
            dt_unit=DT_UNIT,
            K=26,
            measure=measure
        ),
        LinearModel(id=f'{measure}_predictors', predictors=normed_predictors, measure=measure)
    ])

predict_variance = torch.nn.Embedding(
    num_embeddings=len(dataset_all.group_names),
    embedding_dim=len(normed_measures),
    padding_idx=0
)
group_names_to_group_ids = {g: i for i, g in enumerate(dataset_all.group_names)}

kf = KalmanFilter(
    measures=normed_measures,
    processes=processes,
    measure_covariance=Covariance.for_measures(normed_measures, var_predict={'group_ids': predict_variance})
)
optimizer = torch.optim.Adam(kf.parameters(), lr=0.01)

# +
epochs = 350
train_losses = []
val_losses = []


def closure_step():
    optimizer.zero_grad()
    y, X = dataset_train.tensors
    pred = kf(
        y,
        X=X,
        start_datetimes=dataset_train.start_datetimes,
        group_ids=[group_names_to_group_ids[g] for g in dataset_train.group_names]
    )
    loss = -pred.log_prob(y).mean()
    loss.backward()
    return loss


def train():
    return optimizer.step(closure_step).item()


def validate():
    y, X = dataset_val.tensors
    with torch.no_grad():
        pred = kf(
            y,
            X=X,
            start_datetimes=dataset_val.start_datetimes,
            group_ids=[group_names_to_group_ids[g] for g in dataset_train.group_names]
        )
        return -pred.log_prob(y).mean().item()


for i in tqdm(range(epochs)):
    train_loss = train()
    val_loss = validate()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    torch.save(kf.state_dict(), 'cta_model.pt')
# -

fig = plt.figure(figsize=(24, 8))
subplt = plt.subplot(111)
subplt.plot(list(range(epochs)), train_losses, linestyle='-', color=blue)
subplt.plot(list(range(epochs)), val_losses, linestyle='-', color=red)


# ### Visualizations ###

# +
def inverse_transform(df: pd.DataFrame, scalers) -> pd.DataFrame:
    df = df.copy()
    df['measure'] = df['measure'].str.replace('_scaled', '')
    cols_to_transform = ['actual', 'lower', 'mean', 'upper']
    for measure, measure_df in df.groupby('measure'):
        scaler = scalers[measure]
        for col in cols_to_transform:
            new_vals = scaler.inverse_transform(measure_df[col].values.reshape(-1, 1)).flatten()
            df.loc[df['measure'] == measure, col] = new_vals
    return df


y, _ = dataset_train.tensors
_, X = dataset_all.tensors
with torch.no_grad():
    pred = kf(
        y,
        X=X,
        start_datetimes=dataset_train.start_datetimes,
        group_ids=[group_names_to_group_ids[g] for g in dataset_train.group_names],
        out_timesteps=X.shape[1]
    )

df_pred = inverse_transform(pred.to_dataframe(dataset_all), scalers)
df_components = pred.to_dataframe(dataset_all, type='components')

df_pred.to_parquet('cta_predictions.parquet')
df_components.to_parquet('cta_prediction_components.parquet')
