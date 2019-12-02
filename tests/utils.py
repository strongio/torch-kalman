import torch

from torch_kalman.design import Design
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalTrend, Process, FourierSeasonFixed, LocalLevel, Season

import numpy as np


def simulate(num_groups: int, num_timesteps: int, season_spec: dict, noise: float = 1.0) -> torch.Tensor:
    # make kf:
    processes = [
        LocalLevel(id='local_level').add_measure('y'),
        Season(id='day_in_week', seasonal_period=7, fixed=True, **season_spec).add_measure('y'),
        FourierSeasonFixed(id='day_in_month', seasonal_period=30, K=2, **season_spec).add_measure('y')
    ]
    kf = KalmanFilter(measures=['y'], processes=processes)

    # make local-level less aggressive:
    pcov = kf.design.process_covariance.create().data
    pcov[0, 0] *= .1
    kf.design.process_covariance.set(pcov)

    # simulate:
    start_datetimes = np.zeros(num_groups, dtype='timedelta64') + season_spec['season_start']
    with torch.no_grad():
        dfb = kf.design.for_batch(num_groups=num_groups, num_timesteps=num_timesteps, start_datetimes=start_datetimes)
        initial_state = kf.predict_initial_state(dfb)
        simulated_trajectories = initial_state.simulate_trajectories(dfb)
        sim_data = simulated_trajectories.sample_measurements(eps=noise)

    return sim_data


def simple_mv_velocity_design(dims: int = 2):
    processes, measures = [], []
    for i in range(dims):
        process = LocalTrend(id=str(i), decay_velocity=False)
        measure = str(i)
        process.add_measure(measure=measure)
        processes.append(process)
        measures.append(measure)
    return Design(processes=processes, measures=measures)


def name_to_proc(id: str, **kwargs) -> Process:
    season_start = '2010-01-04'

    if 'hour_in_day' in id:
        out = FourierSeasonFixed(id=id,
                                 seasonal_period=24, season_start=season_start, dt_unit='h',
                                 **kwargs)
    elif 'day_in_year' in id:
        out = FourierSeasonFixed(id=id,
                                 seasonal_period=24 * 364.25, season_start=season_start, dt_unit='h',
                                 **kwargs)
    elif 'local_level' in id:
        out = LocalLevel(id=id, **kwargs)
    elif 'local_trend' in id:
        out = LocalTrend(id=id, **kwargs)
    elif 'day_in_week' in id:
        out = Season(
            id=id,
            seasonal_period=7, season_duration=24,
            season_start=season_start, dt_unit='h',
            **kwargs
        )
    else:
        raise NotImplementedError(f"Unsure what process to use for `{id}`.")

    return out
