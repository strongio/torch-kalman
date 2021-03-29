import itertools
from unittest import TestCase

import torch
from parameterized import parameterized
from torch import nn

import numpy as np
from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LinearModel, NN
from torchcast.process.season import FourierSeason


class TestProcess(TestCase):
    def test_fourier_season(self):
        series = torch.sin(2. * 3.1415 * torch.arange(0., 7.) / 7.)
        data = torch.stack([series.roll(-i).repeat(3) for i in range(6)]).unsqueeze(-1)
        start_datetimes = np.array([np.datetime64('2019-04-18') + np.timedelta64(i, 'D') for i in range(6)])
        kf = KalmanFilter(
            processes=[FourierSeason(id='day_of_week', period='7D', dt_unit='D', K=3)],
            measures=['y']
        )
        kf._scale_by_measure_var = False
        kf.state_dict()['processes.day_of_week.init_mean'][:] = torch.tensor([1., 0., 0., 0., 0., 0.])
        kf.state_dict()['measure_covariance.cholesky_log_diag'] -= 2
        pred = kf(data, start_datetimes=start_datetimes)
        for g in range(6):
            self.assertLess(torch.abs(pred.means[g] - data[g]).mean(), .01)

    def test_nn(self):
        y = torch.zeros((2, 5, 1))
        proc = NN(id='nn', nn=nn.Linear(in_features=10, out_features=2, bias=False))
        kf = KalmanFilter(processes=[proc], measures=['y'])
        # TODO

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2, 3]))
    @torch.no_grad()
    def test_lm(self, num_groups: int = 1, num_preds: int = 1):
        data = torch.zeros((num_groups, 5, 1))
        kf = KalmanFilter(
            processes=[
                LinearModel(id='lm', predictors=[f"x{i}" for i in range(num_preds)])
            ],
            measures=['y']
        )
        kf = torch.jit.script(kf)  # TODO: why is this necessary?
        wrong_dim = 1 if num_preds > 1 else 2
        with self.assertRaises((RuntimeError, torch.jit.Error), msg=(num_groups, num_preds)) as cm:
            kf(data, X=torch.zeros((num_groups, 5, wrong_dim)))
        expected = f"produced output with shape [{num_groups}, {wrong_dim}], but expected ({num_preds},) " \
                   f"or (num_groups, {num_preds}). Input had shape [{num_groups}, {wrong_dim}]"
        self.assertIn(expected, str(cm.exception))

        kf(data, X=torch.ones(num_groups, data.shape[1], num_preds))
