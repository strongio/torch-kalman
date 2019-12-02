from itertools import product
from unittest import TestCase

from torch import Tensor

from torch_kalman.kalman_filter import KalmanFilter

import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter

from tests.utils import simple_mv_velocity_design


class TestKalmanFilter(TestCase):

    def test_equations(self):
        data = Tensor([[-50., 50., 1.]])[:, :, None]

        #
        _design = simple_mv_velocity_design(dims=1)
        torch_kf = KalmanFilter(processes=_design.processes.values(), measures=_design.measures)
        batch_design = torch_kf.design.for_batch(1, 1)
        pred = torch_kf(data)

        #
        filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
        filter_kf.x = batch_design.initial_mean.detach().numpy().T
        filter_kf.P = batch_design.initial_covariance.detach().numpy().squeeze(0)

        filter_kf.F = batch_design.F(0)[0].detach().numpy()
        filter_kf.H = batch_design.H(0)[0].detach().numpy()
        filter_kf.R = batch_design.R(0)[0].detach().numpy()
        filter_kf.Q = batch_design.Q(0)[0].detach().numpy()
        filter_kf.states = []
        for t in range(data.shape[1]):
            filter_kf.states.append(filter_kf.x)
            filter_kf.update(data[:, t, :])
            filter_kf.predict()
        filterpy_states = np.stack(filter_kf.states).squeeze()
        kf_states = pred.means.detach().numpy().squeeze()

        for r, c in product(*[range(x) for x in kf_states.shape]):
            self.assertAlmostEqual(filterpy_states[r, c], kf_states[r, c], places=3)
