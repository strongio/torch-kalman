# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: 'Python 3.7.9 64-bit (''torch-kalman'': venv)'
#     metadata:
#       interpreter:
#         hash: e593fb664dc87c5fcb156f9082aabc58b575809650fd778da08a33de5f780190
#     name: python3
# ---

# +
import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sb
sb.set(style='white', color_codes=True)

blue, red, yellow, green, orange, pink, purple, brown = sb.xkcd_palette(
    ['windows blue', 'pale red', 'amber', 'deep green', 'reddish orange', 'pink', 'purple', 'brown']
)

# +
DT_UNIT = 'D'
START_DT = pd.Timestamp('2008-06-01')
END_DT = pd.Timestamp('2018-06-01')
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

color_map = {
    'red': '#c60c30',
    'blue': '#00a1de',
    'green': '#009b3a',
    'brown': '#62361b',
    'pink': '#e27ea6',
    'purple': '#522398',
    'orange': '#f9461c',
    'yellow': '#f9e300'
}

lines = ('red', 'blue', 'green', 'brown')

# +
df_pred = pd.read_parquet('cta_predictions.parquet')
df_pred['time'] = pd.to_datetime(df_pred['time'])
df_pred = df_pred[(df_pred['time'] >= START_DT) & (df_pred['time'] <= END_DT)]

df_components = pd.read_parquet('cta_predictions_components.parquet')
df_components['time'] = pd.to_datetime(df_components['time'])
df_components = df_components[(df_components['time'] >= START_DT) & (df_components['time'] <= END_DT)]

# +
fig, subplts = plt.subplots(nrows=len(lines), ncols=1, sharex=True, figsize=(24 * 3, 12 * 3))
subplts = subplts.flatten()

x_col = 'time'
for group, subplt in zip(lines, subplts):
    df_to_plot = df_pred.query(f"group=='{group}'")
    x = df_to_plot[x_col]
    ypred = df_to_plot['mean']
    yactual = df_to_plot['actual']
    lower = df_to_plot['lower'].clip(lower=0)
    upper = df_to_plot['upper']
    subplt.fill_between(x, y1=lower, y2=upper, color='gray', alpha=0.4)
    subplt.plot(x, ypred, color='k', lw=1, alpha=0.5)
    subplt.plot(x, yactual, color=color_map[group], lw=1, alpha=0.5)
    for yr in np.arange(2009, 2019, 1):
        subplt.axvline(datetime.datetime(yr, 1, 1), linestyle=':', color='k', lw=1)
    subplt.axvline(SPLIT_DT, linestyle='--', color='k')
    subplt.axhline(0, linestyle=':', color='k')
    subplt.set_ylabel(f'{group.capitalize()} Line Daily Riders', fontsize=30, labelpad=20)
    subplt.tick_params(axis='y', labelsize=30)
    subplt.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ymin, ymax = subplt.get_ylim()
    subplt.fill_between([SPLIT_DT, x.iloc[-1]], y1=ymin, y2=ymax, color='k', alpha=0.1, zorder=-2)
    subplt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
subplts[-1].set_xlabel('Date', fontsize=40)
subplts[-1].tick_params(axis='both', labelsize=30)

subplts[0].text(SPLIT_DT + pd.Timedelta(10, 'd'), 300000, u"Validation Data", fontsize=30, ha='left', va='center')
subplts[0].text(SPLIT_DT - pd.Timedelta(10, 'd'), 300000, u"Training Data", fontsize=30, ha='right', va='center')

# +
fig, subplts = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(24, 12))
subplt1, subplt2 = subplts

x_col = 'time'
group = 'blue'
df_to_plot = df_pred.query(
    f"(group=='{group}') & "
    f"(time >= '{(SPLIT_DT - pd.Timedelta(365, unit=DT_UNIT)).strftime('%Y-%m-%d')}') & "
    f"(time <= '{(SPLIT_DT + pd.Timedelta(365, unit=DT_UNIT)).strftime('%Y-%m-%d')}')"
)
x = df_to_plot[x_col]
ypred = df_to_plot['mean']
yactual = df_to_plot['actual']
lower = df_to_plot['lower'].clip(lower=0)
upper = df_to_plot['upper']


subplt1.plot(x, yactual, color=color_map[group], lw=2, alpha=1.0)

subplt1.plot(x, ypred, color='k', lw=1, alpha=1.0)
subplt1.fill_between(x, y1=lower, y2=upper, color='gray', alpha=0.4)

#residuals = ((df_to_plot['actual'] - df_to_plot['mean']) / df_to_plot['actual'])
residuals = (df_to_plot['actual'] - df_to_plot['mean'])
subplt2.plot(x, residuals, 'k-', alpha=1.0)
for subplt in subplts:
    subplt.tick_params(axis='both', labelsize=20)
    subplt.axvline(SPLIT_DT, linestyle='--', color='k')
    subplt.axhline(0, linestyle=':', color='k')
    ymin, ymax = subplt.get_ylim()
    subplt.fill_between([SPLIT_DT, x.iloc[-1]], y1=ymin, y2=ymax, color='k', alpha=0.08, zorder=-2)


val_data = df_to_plot.loc[df_to_plot['time'] >= SPLIT_DT, :]
val_correct_ratio = ((val_data['actual'] < val_data['upper']) & (val_data['actual'] > val_data['lower'])).mean()
rms = np.sqrt(((val_data['actual'] - val_data['mean']) ** 2).mean())
subplt2.text(0.95, 0.94, f'RMS error in validation data: {rms:,.0f}', fontsize=20, ha='right', va='top', transform=subplt2.transAxes)

subplt1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
subplt2.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
subplt1.text(0.51, 0.13, u"Validation Data \u2192", fontsize=20, ha='left', va='top', transform=subplt1.transAxes)
subplt1.text(0.49, 0.13, u"\u2190 Training Data", fontsize=20, ha='right', va='top', transform=subplt1.transAxes)
subplt2.set_xlabel('Date', fontsize=25)

#subplt1.set_title(f'{group.capitalize()} Line Ridership', fontsize=20)
#subplt2.set_title('Residuals (Actual Riders - Predicted Riders)', fontsize=20)
subplt1.text(1.0, 1.0, f'{group.capitalize()} Line Ridership', fontsize=20, ha='right', va='bottom', transform=subplt1.transAxes)
subplt1.text(1.0, 1.0, 'Residuals (Actual Riders - Predicted Riders)', ha='right', va='bottom', fontsize=20, transform=subplt2.transAxes)

# +
fig, subplts = plt.subplots(nrows=2, ncols=len(df_components['process'].unique()) // 2, sharex=True, figsize=(36, 12))
subplts = subplts.flatten()

group = 'red'
df_to_plot = df_components.query(f"(group=='{group}')")
x_col = 'time'
for subplt, (process, dat) in zip(subplts, df_to_plot.groupby('process')):
    for elem, elem_dat in dat.groupby('state_element'):
        x = elem_dat[x_col]
        ypred = elem_dat['mean']
        lower = elem_dat['lower']
        upper = elem_dat['upper']
        subplt.fill_between(x, y1=lower, y2=upper, color='gray', alpha=0.2)
        subplt.plot(x, ypred, color=blue, lw=1)
        subplt.axvline(SPLIT_DT.to_numpy(), linestyle='--', color='k')
    subplt.set_title(process)

# +
import matplotlib.dates as mdates
predictors_df = df_components.query(f"(group=='{group}') & (process.str.endswith('predictors')) & (state_element.str.contains('week') == False)")

fig, subplts = plt.subplots(nrows=len(predictors_df['state_element'].unique()), sharex=True, figsize=(24, 12))
subplt = subplts.flatten()
plt.subplots_adjust(wspace=0.4)

text_map = {'is_cubs_game': 'Cubs Home Game', 'is_holiday': 'Federal Holiday', 'prcp': 'Precipitation', 'tmax': 'Temperature'}
for subplt, (pred, pred_df) in zip(np.flip(subplts), predictors_df.groupby('state_element')):
    x = pred_df[x_col]
    ypred = pred_df['mean']
    lower = pred_df['lower']
    upper = pred_df['upper']
    subplt.plot(x, ypred, color=blue, lw=0.5, label=pred)
    subplt.text(1.0, 1.0, text_map[pred.replace('_scaled', '')], fontsize=15, transform=subplt.transAxes, ha='right', va='bottom')
    subplt.axvline(SPLIT_DT, linestyle='--', color='k')
    for yr in np.arange(2009, 2019, 1):
        subplt.axvline(datetime.datetime(yr, 1, 1), linestyle=':', color='k', lw=1)
    
    subplt.set_ylabel('Relative Change', fontsize=15)
    subplt.tick_params(axis='both', labelsize=15)
    subplt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
subplts[-1].set_xlabel('Date', fontsize=20)

# +
fig = plt.figure(figsize=(24, 8))
subplt = plt.subplot(111)

df_to_plot = df_pred.query("(group == 'red') & (time >= '2012-01-01') & (time < '2016-01-01')")
subplt.plot(df_to_plot['time'], df_to_plot['actual'], color=color_map['red'], zorder=1)
subplt.text(datetime.datetime(2013, 5, 19), 0, 'Red Line South\nModernization Project', ha='left', va='bottom', fontsize=13, zorder=0)

t = np.arange(datetime.datetime(2013,5,19), datetime.datetime(2013,10,20), datetime.timedelta(days=1)).astype(datetime.datetime)
subplt.fill_between(t, y1=0, y2=250000, color='k', alpha=0.1, zorder=-1)

for yr in np.arange(2012, 2016, 1):
    t = np.arange(datetime.datetime(yr,12,15), datetime.datetime(yr+1,1,5), datetime.timedelta(days=1)).astype(datetime.datetime)
    subplt.fill_between(t, y1=0, y2=250000, color='k', alpha=0.1, zorder=-1)
    
for yr in np.arange(2012, 2017, 1):
    subplt.axvline(datetime.datetime(yr, 1, 1), linestyle='--', color='k', zorder=2)

subplt.set_xlabel('Date', fontsize=20)
subplt.set_ylabel('Daily Riders', fontsize=20)
subplt.tick_params(axis='both', labelsize=20)
#subplt.set_title('CTA Red Line Ridership', fontsize=20)
subplt.text(1.0, 1.0, 'CTA Red Line Ridership', fontsize=20, ha='right', va='bottom', transform=subplt.transAxes)
subplt.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

# +
fig = plt.figure()
subplt = plt.subplot(111)

subplt.plot(np.arange(1, 12, 1), np.arange(1, 12, 1), linestyle='-', color='#c60c30')
# -


