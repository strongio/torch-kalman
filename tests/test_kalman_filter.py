import copy
from itertools import product
from unittest import TestCase

import torch
from parameterized import parameterized

from torch import Tensor

from torch_kalman.kalman_filter import KalmanFilter

import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter

from tests.utils import simple_mv_velocity_design


class TestKalmanFilter(TestCase):

    @parameterized.expand([(1,), (2,), (3,)])
    def test_nstep_preds(self, n_step: int):
        from torch_kalman.process import LinearModel
        from torch_kalman.utils.data import TimeSeriesDataset
        from pandas import DataFrame

        class LinearModelFixed(LinearModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, process_variance=False)

            @property
            def fixed_state_elements(self):
                return self.state_elements

        kf = KalmanFilter(
            processes=[
                LinearModelFixed(id='lm', covariates=['x1', 'x2']).add_measure('y')
            ],
            measures=['y']
        )
        kf.state_dict()['design_parameters.process:lm.initial_state_mean'][0] = 1.5
        kf.state_dict()['design_parameters.process:lm.initial_state_mean'][1] = -0.5
        kf.state_dict()['design_parameters.measure_cov.cholesky_log_diag'][0] = np.log(.1 ** .5)

        num_times = 100
        df = DataFrame({'x1': np.random.randn(num_times), 'x2': np.random.randn(num_times)})
        df['y'] = 1.5 * df['x1'] + -.5 * df['x2'] + .1 * np.random.randn(num_times)
        df['time'] = df.index.values
        df['group'] = '1'
        dataset = TimeSeriesDataset.from_dataframe(
            dataframe=df,
            group_colname='group',
            time_colname='time',
            dt_unit=None,
            X_colnames=['x1', 'x2'],
            y_colnames=['y']
        )
        y, X = dataset.tensors

        with torch.no_grad():
            pred = kf(y, predictors=X, out_timesteps=X.shape[1] - n_step + 1, n_step=n_step)
            resid = y - pred.predictions
            self.assertLess((resid ** 2).mean(), .02)
            self.assertLess(resid.mean().abs(), .05)

    @parameterized.expand([(1,), (2,), (3,)])
    def test_nstep(self, n_step: int):

        data = Tensor([[-1., 2., 1., 0.]])[:, :, None]
        num_times = data.shape[1]

        # make torch kf:
        _design = simple_mv_velocity_design(dims=1)
        torch_kf = KalmanFilter(processes=_design.processes.values(), measures=_design.measures)
        batch_design = torch_kf.design.for_batch(1, 1)
        pred = torch_kf(data, out_timesteps=num_times - n_step + 1, n_step=n_step)
        kf_states = pred.means.detach().numpy().squeeze()
        self.assertTrue((pred.predictions == pred.means[:, :, [0]]).all())

        # make filterpy kf:
        filter_kf = self._make_filter_kf(batch_design)
        filterpy_states = [filter_kf.x.copy() for _ in range(n_step)]

        # compare:
        for t in range(num_times - n_step):
            filter_kf.update(data[:, t, :])
            # 1step:
            filter_kf.predict()
            # n_step:
            filter_kf_copy = copy.deepcopy(filter_kf)
            for i in range(n_step - 1):
                filter_kf_copy.predict()
            filterpy_states.append(filter_kf_copy.x)

        filterpy_states = np.stack(filterpy_states).squeeze()

        for r, c in product(*[range(x) for x in kf_states.shape]):
            self.assertAlmostEqual(filterpy_states[r, c], kf_states[r, c], places=3)

    def test_equations(self):
        data = Tensor([[-50., 50., 1.]])[:, :, None]

        # make torch kf:
        _design = simple_mv_velocity_design(dims=1)
        torch_kf = KalmanFilter(processes=_design.processes.values(), measures=_design.measures)
        batch_design = torch_kf.design.for_batch(1, 1)
        pred = torch_kf(data)

        # make filterpy kf:
        filter_kf = self._make_filter_kf(batch_design)
        filter_kf.states = []

        # compare:
        for t in range(data.shape[1]):
            filter_kf.states.append(filter_kf.x)
            filter_kf.update(data[:, t, :])
            filter_kf.predict()
        filterpy_states = np.stack(filter_kf.states).squeeze()
        kf_states = pred.means.detach().numpy().squeeze()

        for r, c in product(*[range(x) for x in kf_states.shape]):
            self.assertAlmostEqual(filterpy_states[r, c], kf_states[r, c], places=3)

    def _make_filter_kf(self, batch_design):
        filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
        filter_kf.x = batch_design.initial_mean.detach().numpy().T
        filter_kf.P = batch_design.initial_covariance.detach().numpy().squeeze(0)

        filter_kf.F = batch_design.F(0)[0].detach().numpy()
        filter_kf.H = batch_design.H(0)[0].detach().numpy()
        filter_kf.R = batch_design.R(0)[0].detach().numpy()
        filter_kf.Q = batch_design.Q(0)[0].detach().numpy()
        return filter_kf
