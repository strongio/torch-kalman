import unittest
import numpy as np
import torch
from torch.optim import LBFGS

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import FourierSeasonDynamic, LocalLevel, Season
from tests.utils import simulate


class TestTraining(unittest.TestCase):
    config = {
        'num_groups': 4,
        'num_timesteps': int(30 * 2.5),
        'season_spec': {'season_start': np.datetime64('2018-01-01'), 'dt_unit': 'D'}
    }

    def _train_kf(self, data: torch.Tensor, num_epochs: int = 8):
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
            pred = kf(data, start_datetimes=start_datetimes)
            loss = -pred.log_prob(data).mean()
            loss.backward()
            return loss

        print(f"Will train for {num_epochs} epochs...")
        loss = float('nan')
        for i in range(num_epochs):
            new_loss = kf.opt.step(closure)
            print(f"EPOCH {i}, LOSS {new_loss.item()}, DELTA {loss - new_loss.item()}")
            loss = new_loss.item()

        return kf(data, start_datetimes=start_datetimes).predictions

    def test_training_from_sim(self):
        sim_data = simulate(**self.config, noise=0.1)
        predictions = self._train_kf(sim_data)
        se = (sim_data - predictions) ** 2
        self.assertLess(se.mean().item(), .4)

    def test_training_from_manual_sim(self):
        sim_data = (
                torch.cumsum(torch.randn(self.config['num_timesteps']), 0) +
                torch.sin(torch.arange(1., self.config['num_timesteps'] + 1., 1.) / 10.)
        )
        sim_data = sim_data[None, :, None] + torch.randn((self.config['num_groups'], self.config['num_timesteps'], 1))
        predictions = self._train_kf(sim_data, num_epochs=5)
        se = (sim_data - predictions) ** 2
        self.assertLess(se.mean().item(), 5.0)
