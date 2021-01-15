import time
import unittest

import numpy as np
import torch
from parameterized import parameterized

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, LinearModel


class TestTraining(unittest.TestCase):
    @parameterized.expand([(1,), (2,), (3,)])
    @torch.no_grad()
    def test_gaussian_log_prob(self, ndim: int = 1):
        data = torch.zeros((2, 5, ndim))
        kf = KalmanFilter(
            processes=[LocalLevel(id=f'lm{i}', measure=str(i)) for i in range(ndim)],
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
            processes=[LocalLevel(id=f'lm{i}', measure=str(i)) for i in range(ndim)],
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

    def test_training1(self, ndim: int = 2, num_groups: int = 150, num_times: int = 24):
        """
        simulated data with known parameters, fitted loss should approach the loss given known params
        """
        torch.manual_seed(123)

        def _make_kf():
            return KalmanFilter(
                processes=[
                              LocalLevel(id=f'll{i}', measure=str(i))
                              for i in range(ndim)
                          ] + [
                              LinearModel(id=f'lm{i}', predictors=['x1', 'x2', 'x3', 'x4', 'x5'], measure=str(i))
                              for i in range(ndim)
                          ],
                measures=[str(i) for i in range(ndim)]
            )

        # simulate:
        X = torch.randn((num_groups, num_times, 5))
        kf_generator = _make_kf()
        with torch.no_grad():
            kf_generator.state_dict()['script_module.process_covariance.cholesky_log_diag'] -= 2.
            sim = kf_generator.simulate(out_timesteps=num_times, num_sims=num_groups, X=X)
            y = sim.sample()
        assert not y.requires_grad

        # train:
        kf_learner = _make_kf()
        optimizer = torch.optim.LBFGS(kf_learner.parameters())
        times = []

        def closure():
            optimizer.zero_grad()
            _start = time.time()
            pred = kf_learner(y, X=X)
            times.append(time.time() - _start)
            loss = -pred.log_prob(y).mean()
            loss.backward()
            # print(loss.item())
            return loss

        print("\nTraining for 5 epochs...")
        for i in range(5):
            loss = optimizer.step(closure)
            print("loss:", loss.item())

        print("time:", np.nanmean(times))

        oracle_loss = -kf_generator(y, X=X).log_prob(y).mean()
        self.assertAlmostEqual(oracle_loss.item(), loss.item(), places=1)

    def test_training2(self, ndim: int = 2, num_groups: int = 10, num_times: int = 50):
        """
        # manually generated data (sin-wave, trend, etc.) with virtually no noise: MSE should be near zero
        """
        pass  # TODO
