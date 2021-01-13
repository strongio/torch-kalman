import copy
from collections import defaultdict
from typing import Callable
from unittest import TestCase
import mock

import torch
from parameterized import parameterized

from torch import nn

from torch_kalman.kalman_filter import KalmanFilter

import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter

from torch_kalman.process import LocalTrend, LinearModel
from torch_kalman.process.base import Process
from torch_kalman.process.utils import SingleOutput


class TestKalmanFilter(TestCase):
    @torch.no_grad()
    def test_jit(self):
        from torch_kalman.kalman_filter.state_belief_over_time import StateBeliefOverTime

        # compile-able:
        h_module = SingleOutput()
        f_modules = torch.nn.ModuleDict()
        f_modules['position->position'] = SingleOutput()

        compilable = Process(id='compilable',
                             state_elements=['position'],
                             h_module=h_module,
                             f_modules=f_modules)

        torch_kf = KalmanFilter(
            processes=[compilable],
            measures=['y']
        )
        # runs:
        self.assertIsInstance(torch_kf(torch.tensor([[[-5., 5., 1.]]])), StateBeliefOverTime)

        # not compile-able:
        not_compilable = Process(id='not_compilable',
                                 state_elements=['position'],
                                 h_module=lambda x=None: h_module(x),
                                 f_modules={k: lambda x=None: v(x) for k, v in f_modules.items()})
        with self.assertRaises(RuntimeError) as cm:
            torch_kf = KalmanFilter(
                processes=[not_compilable],
                measures=['y']
            )
        the_exception = cm.exception
        self.assertIn('failed to compile', str(the_exception))
        self.assertIn('TorchScript', str(the_exception))

        # but we can skip compilation:
        torch_kf = KalmanFilter(
            processes=[not_compilable],
            measures=['y'],
            compiled=False
        )
        self.assertIsInstance(torch_kf(torch.tensor([[[-5., 5., 1.]]])), StateBeliefOverTime)

    @parameterized.expand([(0,), (1,), (2,), (3,)])
    @torch.no_grad()
    def test_equations(self, n_step: int):
        data = torch.tensor([[-5., 5., 1., 0., 3.]]).unsqueeze(-1)
        num_times = data.shape[1]

        # make torch kf:
        torch_kf = KalmanFilter(
            processes=[LocalTrend(id='lt', decay_velocity=None).set_measure('y')],
            measures=['y'],
            compiled=n_step > 0
        )
        expectedF = torch.tensor([[1., 1.], [0., 1.]])
        expectedH = torch.tensor([[1., 0.]])
        _kwargs = torch_kf._parse_design_kwargs(input=data, out_timesteps=num_times)
        init_mean_kwargs = _kwargs.pop('init_mean_kwargs')
        design_kwargs = torch_kf.script_module._get_design_kwargs_for_time(time=0, **_kwargs)
        F, H, Q, R = torch_kf.script_module.get_design_mats(num_groups=1, design_kwargs=design_kwargs)
        assert torch.isclose(expectedF, F).all()
        assert torch.isclose(expectedH, H).all()

        # make filterpy kf:
        filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
        filter_kf.x, filter_kf.P = torch_kf.script_module.get_initial_state(data, init_mean_kwargs, {})
        filter_kf.x = filter_kf.x.detach().numpy().T
        filter_kf.P = filter_kf.P.detach().numpy().squeeze(0)
        filter_kf.Q = Q.numpy().squeeze(0)
        filter_kf.R = R.numpy().squeeze(0)
        filter_kf.F = F.numpy().squeeze(0)
        filter_kf.H = H.numpy().squeeze(0)

        # compare:
        if n_step == 0:
            with self.assertRaises(AssertionError):
                torch_kf(data, n_step=n_step)
            return
        else:
            sb = torch_kf(data, n_step=n_step)

        #
        filter_kf.means = []
        filter_kf.covs = []
        for t in range(num_times):
            if t >= n_step:
                filter_kf.update(data[:, t - n_step, :])
                # 1step:
                filter_kf.predict()
            # n_step:
            filter_kf_copy = copy.deepcopy(filter_kf)
            for i in range(1, n_step):
                filter_kf_copy.predict()
            filter_kf.means.append(filter_kf_copy.x)
            filter_kf.covs.append(filter_kf_copy.P)

        assert np.isclose(sb.means.numpy().squeeze(), np.stack(filter_kf.means).squeeze()).all()
        assert np.isclose(sb.covs.numpy().squeeze(), np.stack(filter_kf.covs).squeeze()).all()

    @parameterized.expand([(1,), (2,), (3,)])
    @torch.no_grad()
    def test_equations_preds(self, n_step: int):
        from torch_kalman.utils.data import TimeSeriesDataset
        from pandas import DataFrame

        class LinearModelFixed(LinearModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.no_icov_state_elements = self.state_elements

        kf = KalmanFilter(
            processes=[
                LinearModelFixed(id='lm', predictors=['x1', 'x2'])
            ],
            measures=['y'],
            compiled=False
        )
        kf.state_dict()['script_module.processes.0.init_mean'][:] = torch.tensor([1.5, -0.5])
        kf.state_dict()['script_module.measure_covariance.cholesky_log_diag'][0] = np.log(.1 ** .5)

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

        from pandas import Series

        pred = kf(y, predictors=X, out_timesteps=X.shape[1], n_step=n_step)
        y_series = Series(y.squeeze().numpy())
        for shift in range(-2, 3):
            resid = y_series.shift(shift) - Series(pred.predictions.squeeze().numpy())
            if shift:
                # check there's no misalignment in internal n_step logic (i.e., realigning the input makes things worse)
                self.assertGreater((resid ** 2).mean(), 1.)
            else:
                self.assertLess((resid ** 2).mean(), .02)

    @mock.patch(f'{Process.__module__}.Process.f_forward')
    def test_process_caching(self, mock_f_forward: mock.Mock):
        data = torch.tensor([[-5., 5., 1., 0., 3.]]).unsqueeze(-1)
        mock_f_forward.return_value = torch.tensor([[1., 0.], [1., 0.]])
        torch_kf = KalmanFilter(
            processes=[LocalTrend(id='lt').set_measure('y')],
            measures=['y'],
            compiled=False
        )
        # with cache enabled, only called once
        torch_kf(data)
        self.assertEqual(mock_f_forward.call_count, 1)

        # which is less than what we'd expect without the cache, which is data.shape[1] times
        torch_kf.script_module.processes[0].cache.clear()
        torch_kf.script_module.processes[0].enable_cache = lambda enable: None
        torch_kf(data)
        self.assertEqual(mock_f_forward.call_count, 1 + data.shape[1])

    def test_keyword_dispatch(self):
        _counter = defaultdict(int)

        def check_input(func: Callable, expected: torch.Tensor) -> Callable:
            def outfunc(x):
                _counter[func.__name__] += 1
                self.assertIsNotNone(x)
                self.assertTrue((x == expected).all().item())
                return func(x)

            return outfunc

        data = torch.tensor([[-5., 5., 1., 0., 3.]]).unsqueeze(-1)

        def _make_kf():
            return KalmanFilter(
                processes=[
                    LinearModel(id='lm1', predictors=['x1', 'x2']),
                    LinearModel(id='lm2', predictors=['x1', 'x2'])
                ],
                measures=['y'],
                compiled=False
            )

        _predictors = torch.ones(1, data.shape[1], 2)

        # shared --
        expected = {'lm1': torch.zeros(1), 'lm2': torch.zeros(1)}
        # share input:
        kf = _make_kf()
        for nm, proc in kf.named_processes():
            proc.h_forward = check_input(proc.h_forward, expected[nm])
        kf(data, predictors=_predictors * 0.)

        # separate ---
        expected['lm2'] = torch.ones(1)
        # individual input:
        kf = _make_kf()
        for nm, proc in kf.named_processes():
            proc.h_forward = check_input(proc.h_forward, expected[nm])
        kf(data, lm1__predictors=_predictors * 0., lm2__predictors=_predictors)

        # specific overrides general
        kf(data, predictors=_predictors * 0., lm2__predictors=_predictors)

        # make sure check_input is being called:
        self.assertEqual(_counter['h_forward'] / data.shape[1] / 2, 3)
        with self.assertRaises(AssertionError) as cm:
            kf(data, predictors=_predictors * 0.)
        self.assertEqual(str(cm.exception).lower(), "false is not true")

    def test_current_time(self):
        _state = {}

        def make_season(current_time: torch.Tensor):
            self.assertEqual(current_time, _state['call_counter'])
            _state['call_counter'] += 1
            return current_time % 7

        class Season(Process):
            def __init__(self, id: str):
                super(Season, self).__init__(
                    id=id,
                    h_module=make_season,
                    state_elements=['x'],
                    f_tensors={'x->x': torch.ones(1)}
                )
                self.h_kwarg = 'current_time'
                self.time_varying_kwargs = ['current_time']

        torch_kf = KalmanFilter(
            processes=[Season(id='s1')],
            measures=['y'],
            compiled=False
        )
        data = torch.arange(7).view(1, -1, 1)
        for init_state in [0., 1.]:
            torch_kf.state_dict()['script_module.processes.0.init_mean'][:] = torch.ones(1) * init_state
            _state['call_counter'] = 0
            pred = torch_kf(data)
            # make sure test was called each time:
            self.assertEqual(_state['call_counter'], data.shape[1])

            # more suited to a season test but we'll check anyways:
            if init_state == 1.:
                self.assertTrue((pred.means == 1.).all())
            else:
                self.assertGreater(pred.means[:, -1], pred.means[:, 0])
