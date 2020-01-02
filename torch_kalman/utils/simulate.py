import torch

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, Season, FourierSeasonFixed

import numpy as np

from torch_kalman.utils.data import TimeSeriesDataset


def _simulate(num_groups: int, num_timesteps: int, season_spec: dict, noise: float = 1.0) -> torch.Tensor:
    # make kf:
    processes = [
        LocalLevel(id='local_level').add_measure('y'),
        Season(id='day_in_week', seasonal_period=7, fixed=True, **season_spec).add_measure('y'),
        FourierSeasonFixed(id='day_in_year', seasonal_period=365.25, K=2, **season_spec).add_measure('y')
    ]
    kf = KalmanFilter(measures=['y'], processes=processes)

    # simulate:
    start_datetimes = np.zeros(num_groups, dtype='timedelta64') + season_spec['season_start']
    with torch.no_grad():
        dfb = kf.design.for_batch(num_groups=num_groups, num_timesteps=num_timesteps, start_datetimes=start_datetimes)
        initial_state = kf.predict_initial_state(dfb)
        simulated_trajectories = initial_state.simulate_trajectories(dfb)
        sim_data = simulated_trajectories.sample_measurements(eps=noise)

    return sim_data


def simulate_daily_series(num_groups: int, num_timesteps: int, noise: float = 1.0) -> 'DataFrame':
    season_spec = {
        'season_start': np.datetime64('2007-01-01'),  # arbitrary monday at midnight
        'dt_unit': 'D'
    }
    tensor = _simulate(num_groups, num_timesteps, season_spec, noise=noise)
    dataset = TimeSeriesDataset(
        tensor,
        group_names=range(num_groups),
        start_times=[season_spec['season_start']] * num_groups,
        measures=[['y']],
        dt_unit=season_spec['dt_unit']
    )
    df = dataset.to_dataframe()
    # TODO: meaningful predictors
    df['X1'] = np.random.normal(size=len(df.index))
    df['X2'] = np.random.normal(size=len(df.index))
    return df
