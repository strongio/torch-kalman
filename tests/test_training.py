import time
import unittest

import numpy as np
import torch
from parameterized import parameterized

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, LinearModel, LocalTrend
from torch_kalman.process.season import FourierSeason


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
        mask = torch.randn_like(data) > 1.
        while mask.all() or not mask.any():
            mask = torch.randn_like(data) > 1.
        data[mask.nonzero(as_tuple=True)] = float('nan')
        kf = KalmanFilter(
            processes=[LocalTrend(id=f'lm{i}', measure=str(i)) for i in range(ndim)],
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
                if isvalid_gt.all():
                    lp_gt = kf.kf_step.log_prob(data_gt, *pred_gt).item()
                else:
                    pred_gtm = pred_gt.observe(
                        state_means=pred_gt.state_means,
                        state_covs=pred_gt.state_covs,
                        R=pred_gt.R[..., isvalid_gt, :][..., isvalid_gt],
                        H=pred_gt.H[..., isvalid_gt, :]
                    )
                    lp_gt = kf.kf_step.log_prob(data_gt[..., isvalid_gt], *pred_gtm).item()
                self.assertAlmostEqual(lp_method1[g, t].item(), lp_gt, places=4)
                lp_method2_sum += lp_gt
        self.assertAlmostEqual(lp_method1_sum, lp_method2_sum, places=3)

    def test_training1(self, ndim: int = 2, num_groups: int = 150, num_times: int = 24):
        """
        simulated data with known parameters, fitted loss should approach the loss given known params
        """
        torch.manual_seed(123)

        # TODO: include nans; make sure performance doesn't take significant hit w/partial nans

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
        optimizer = torch.optim.LBFGS(kf_learner.parameters(), max_iter=10)
        forward_times = []
        backward_times = []

        def closure():
            optimizer.zero_grad()
            _start = time.time()
            # print(f'[{datetime.datetime.now().time()}] forward...')
            pred = kf_learner(y, X=X)
            forward_times.append(time.time() - _start)
            loss = -pred.log_prob(y).mean()
            _start = time.time()
            # print(f'[{datetime.datetime.now().time()}] backward...')
            loss.backward()
            backward_times.append(time.time() - _start)
            # print(f'[{datetime.datetime.now().time()}] {loss.item()}')
            return loss

        print("\nTraining for 5 epochs...")
        for i in range(5):
            loss = optimizer.step(closure)
            print("loss:", loss.item())

        # print("forward time:", np.nanmean(forward_times))
        # print("backward time:", np.nanmean(backward_times))

        oracle_loss = -kf_generator(y, X=X).log_prob(y).mean()
        self.assertAlmostEqual(oracle_loss.item(), loss.item(), places=1)

    def test_training2(self, num_groups: int = 50):
        """
        # manually generated data (sin-wave, trend, etc.) with virtually no noise: MSE should be near zero
        """
        weekly = torch.sin(2. * 3.1415 * torch.arange(0., 7.) / 7.)
        data = torch.stack([
            weekly.roll(-i).repeat(3) + torch.linspace(0, 10, 7 * 3) for i in range(num_groups)
        ]).unsqueeze(-1)
        start_datetimes = np.array([np.datetime64('2019-04-14') + np.timedelta64(i, 'D') for i in range(num_groups)])
        kf = KalmanFilter(
            processes=[
                LocalTrend(id='trend'),
                FourierSeason(id='day_of_week', period=7, dt_unit='D', K=3)
            ],
            measures=['y']
        )

        # train:
        kf.state_dict()['script_module.measure_covariance.cholesky_log_diag'] -= 1
        optimizer = torch.optim.LBFGS([p for n, p in kf.named_parameters() if 'measure_covariance' not in n],
                                      lr=.25,
                                      max_iter=10)

        def closure():
            optimizer.zero_grad()
            _start = time.time()
            # print(f'[{datetime.datetime.now().time()}] forward...')
            pred = kf(data, start_datetimes=start_datetimes)
            loss = -pred.log_prob(data).mean()
            _start = time.time()
            loss.backward()
            return loss

        print("\nTraining for 10 epochs...")
        for i in range(10):
            loss = optimizer.step(closure)
            print("loss:", loss.item())

        pred = kf(data, start_datetimes=start_datetimes)
        # MSE should be virtually zero
        self.assertLess(torch.mean((pred.means - data) ** 2), .01)
        # trend should be identified:
        self.assertAlmostEqual(pred.state_means[:, :, 1].mean().item(), 5., places=1)
