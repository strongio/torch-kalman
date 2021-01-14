import unittest
from typing import Type

import numpy as np
import torch
from parameterized import parameterized

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel


class TestTraining(unittest.TestCase):
    @parameterized.expand([(1,), (2,), (3,)])
    @torch.no_grad()
    def test_gaussian_log_prob(self, ndim: int = 1):
        data = torch.zeros((2, 5, ndim))
        kf = KalmanFilter(
            processes=[LocalLevel(id=f'lm{i}').set_measure(str(i)) for i in range(ndim)],
            measures=[str(i) for i in range(ndim)],
            compiled=False
        )
        pred = kf(data)
        log_lik1 = kf.kf_step.log_prob(data, *pred)
        from torch.distributions import MultivariateNormal
        mv = MultivariateNormal(*pred)
        log_lik2 = mv.log_prob(data)
        self.assertAlmostEqual(log_lik1.sum().item(), log_lik2.sum().item())

    def test_log_prob_with_missings(self, ndim: int = 2):
        data = torch.zeros((2, 5, ndim))
        data[0, 1, :] = float('nan')
        kf = KalmanFilter(
            processes=[LocalLevel(id=f'lm{i}').set_measure(str(i)) for i in range(ndim)],
            measures=[str(i) for i in range(ndim)],
            compiled=False
        )
        pred = kf(data)
        print(pred.log_prob(data))


class Foo:
    config = {
        'num_groups': 4,
        'num_timesteps': int(30 * 2.5),
        'dt_unit': 'D'
    }

    def _train_kf(self, data: torch.Tensor, num_epochs: int = 8, cls: Type['KalmanFilter'] = KalmanFilter):
        kf = cls(
            measures=['y'],
            processes=[
                LocalLevel(id='local_level').add_measure('y'),
                Season(id='day_in_week', seasonal_period=7, dt_unit='D').add_measure('y')
            ]
        )
        kf.opt = LBFGS(kf.parameters())

        start_datetimes = (
                np.zeros(self.config['num_groups'], dtype='timedelta64') + DEFAULT_START_DT
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
        sim_data = _simulate(**self.config, noise=0.1)
        predictions = self._train_kf(sim_data)
        se = (sim_data - predictions) ** 2
        self.assertLess(se.mean().item(), .4)

    def test_tobit_training(self):
        class TobitFilter(KalmanFilter):
            family = CensoredGaussian

        sim_data = _simulate(**self.config, noise=0.1)
        predictions = self._train_kf(sim_data, cls=TobitFilter, num_epochs=5)
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
