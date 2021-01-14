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
            measures=[str(i) for i in range(ndim)]
        )
        pred = kf(data)
        log_lik1 = kf.kf_step.log_prob(data, *pred)
        from torch.distributions import MultivariateNormal
        mv = MultivariateNormal(*pred)
        log_lik2 = mv.log_prob(data)
        self.assertAlmostEqual(log_lik1.sum().item(), log_lik2.sum().item())

    @parameterized.expand([(1,), (2,), (3,)])
    @torch.no_grad()
    def test_log_prob_with_missings(self, ndim: int = 1, num_groups: int = 1, num_times: int = 5):
        data = torch.randn((num_groups, num_times, ndim))
        mask = torch.randn((num_groups, num_times)) > 1.
        while mask.all() or not mask.any():
            mask = torch.randn((num_groups, num_times)) > 1.
        data[mask.nonzero(as_tuple=True)] = float('nan')
        kf = KalmanFilter(
            processes=[LocalLevel(id=f'lm{i}').set_measure(str(i)) for i in range(ndim)],
            measures=[str(i) for i in range(ndim)]
        )
        pred = kf(data)
        lp_method1 = pred.log_prob(data)
        lp_method1_sum = lp_method1.sum().item()

        lp_method2_sum = 0
        for g in range(num_groups):
            data_g = data[[g]]
            pred_g = kf(data_g)
            for t in range(num_times):
                pred_gt = pred_g[:, [t]]
                data_gt = data_g[:, [t]]
                isvalid_gt = ~torch.isnan(data_gt).squeeze(0).squeeze(0)
                if not isvalid_gt.any():
                    continue
                isvalid_gt = isvalid_gt.nonzero(as_tuple=False).squeeze(-1)
                lp_gt = kf.kf_step.log_prob(data_gt[..., isvalid_gt], *pred_gt[..., isvalid_gt]).item()
                self.assertAlmostEqual(lp_method1[g, t].item(), lp_gt, places=4)
                lp_method2_sum += lp_gt
        self.assertAlmostEqual(lp_method1_sum, lp_method2_sum, places=3)

    def test_training1(self, ndim: int = 2, num_groups: int = 10, num_times: int = 50):
        """
        simulated data with noise : MSE should get to pre-specified level
        """
        kf_generator = KalmanFilter(
            processes=[LocalLevel(id=f'lm{i}').set_measure(str(i)) for i in range(ndim)],
            measures=[str(i) for i in range(ndim)]
        )
        sim = kf_generator.simulate(out_timesteps=num_times, num_groups=num_groups)
        data = sim.sample()

        kf_learner = KalmanFilter(
            processes=[LocalLevel(id=f'lm{i}').set_measure(str(i)) for i in range(ndim)],
            measures=[str(i) for i in range(ndim)]
        )

        optimizer = torch.optim.LBFGS(kf_learner.parameters(), lr=.25)

        def closure():
            optimizer.zero_grad()
            loss = - kf_learner(data).log_prob(data).mean()
            loss.backward()
            return loss

        for i in range(10):
            loss = optimizer.step(closure)
            print(loss.item())

    def test_training2(self, ndim: int = 2, num_groups: int = 10, num_times: int = 50):
        """
        # manually generated data (sin-wave, trend, etc.) with virtually no noise: MSE should be near zero
        """
        pass  # TODO


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
