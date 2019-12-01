import unittest
import numpy as np
from torch.optim import LBFGS

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import FourierSeasonDynamic, LocalLevel, Season
from torch_kalman.tests.utils import simulate


class TestTraining(unittest.TestCase):
    config = {
        'num_groups': 4,
        'num_timesteps': int(30 * 2.5),
        'season_spec': {'season_start': np.datetime64('2018-01-01'), 'dt_unit': 'D'}
    }

    def test_training(self):
        sim_data = simulate(**self.config, noise=0.1)

        kf = KalmanFilter(
            measures=['y'],
            processes=[
                LocalLevel(id='local_level').add_measure('y'),
                Season(id='day_in_week', seasonal_period=7, **self.config['season_spec']).add_measure('y'),
                FourierSeasonDynamic(
                    id='day_in_month', seasonal_period=30, K=2, **self.config['season_spec']
                ).add_measure('y')
            ]
        )
        kf.opt = LBFGS(kf.parameters())

        start_datetimes = (
                np.zeros(self.config['num_groups'], dtype='timedelta64') + self.config['season_spec']['season_start']
        )

        def closure():
            kf.opt.zero_grad()
            pred = kf(sim_data, start_datetimes=start_datetimes)
            loss = -pred.log_prob(sim_data).mean()
            loss.backward()
            return loss

        print(f"{type(self).__name__} starting training...")
        loss = float('nan')
        for i in range(8):
            new_loss = kf.opt.step(closure)
            print(f"EPOCH {i}, LOSS {new_loss.item()}, DELTA {loss - new_loss.item()}")
            loss = new_loss.item()

        se = (sim_data - kf(sim_data, start_datetimes=start_datetimes).predictions) ** 2
        self.assertLess(se.mean().item(), .4)
