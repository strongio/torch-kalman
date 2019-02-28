from itertools import product

from torch import Tensor

from torch_kalman.covariance import Covariance
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.tests import TestCaseTK, simple_mv_velocity_design, name_to_proc

import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter


class TestKalmanFilter(TestCaseTK):
    season_start = '2010-01-04'

    def test_complex_kf_init(self):
        proc_specs = {'hour_in_day': {'K': 3},
                      'day_in_year': {'K': 3},
                      'local_level': {'decay': (.33, .95)},
                      'local_trend': {'decay_position': (0.95, 1.00), 'decay_velocity': (0.90, 1.00)}
                      }
        processes = []
        for id, pkwargs in proc_specs.items():
            processes.append(name_to_proc(id, **pkwargs))
            processes[-1].add_measure('measure')

        kf = KalmanFilter(measures=['measure'], processes=processes)

    def test_equations(self):
        data = Tensor([[-50., 50., 1.]])[:, :, None]

        #
        design = simple_mv_velocity_design(dims=1)
        batch_design = design.for_batch(1, 1)
        torch_kf = KalmanFilter(processes=design.processes.values(), measures=design.measures)
        pred = torch_kf(data)

        #
        filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
        filter_kf.x = torch_kf.design.init_state_mean_params.detach().numpy()[:, None]
        filter_kf.P = Covariance.from_log_cholesky(torch_kf.design.init_cholesky_log_diag,
                                                   torch_kf.design.init_cholesky_off_diag).detach().numpy()

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
