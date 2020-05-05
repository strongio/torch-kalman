import torch

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, Season, FourierSeason

import numpy as np

from torch_kalman.utils.data import TimeSeriesDataset
from torch_kalman.utils.datetime import DEFAULT_START_DT


def _simulate(num_groups: int, num_timesteps: int, dt_unit: str, noise: float = 1.0) -> torch.Tensor:
    # make kf:
    processes = [
        LocalLevel(id='local_level').add_measure('y'),
        Season(id='day_in_week', seasonal_period=7, fixed=True, dt_unit=dt_unit).add_measure('y'),
        FourierSeason(id='day_in_year', seasonal_period=365.25, K=2, fixed=True, dt_unit=dt_unit).add_measure('y')
    ]
    kf = KalmanFilter(measures=['y'], processes=processes)

    # simulate:
    start_datetimes = np.zeros(num_groups, dtype='timedelta64') + DEFAULT_START_DT
    with torch.no_grad():
        dfb = kf.design.for_batch(num_groups=num_groups, num_timesteps=num_timesteps, start_datetimes=start_datetimes)
        initial_state = kf._predict_initial_state(dfb)
        simulated_trajectories = initial_state.simulate_trajectories(dfb)
        sim_data = simulated_trajectories.sample_measurements(eps=noise)

    return sim_data


def simulate_daily_series(num_groups: int, num_timesteps: int, noise: float = 1.0) -> 'DataFrame':
    # create realistic series:
    tensor = _simulate(num_groups, num_timesteps, noise=noise, dt_unit='D')

    # convert to dataset:
    dataset = TimeSeriesDataset(
        tensor,
        group_names=range(num_groups),
        start_times=[DEFAULT_START_DT] * num_groups,
        measures=[['y']],
        dt_unit='D'
    )
    # convert to dataframe:
    df = dataset.to_dataframe()

    # add predictors:
    # TODO: meaningful predictors
    df['X1'] = np.random.normal(size=len(df.index))
    df['X2'] = np.random.normal(size=len(df.index))

    # make number of timesteps per group non-uniform:
    max_timestep_per_group = dict(zip(
        range(num_groups),
        np.random.choice(range(int(num_timesteps * .80), num_timesteps), size=num_groups, replace=True)
    ))
    df['_max_time'] = DEFAULT_START_DT + df['group'].map(max_timestep_per_group)
    df = df.loc[df['time'] <= df.pop('_max_time'), :].reset_index(drop=True)

    return df
