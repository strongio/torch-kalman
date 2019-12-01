import unittest

import torch
from numpy.linalg import cholesky
from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.process.processes.local_trend import LocalTrend

from torch_kalman.tests.utils import simple_mv_velocity_design
import numpy as np


class TestDesign(unittest.TestCase):
    def test_design_attrs(self):
        with self.assertRaises(ValueError) as cm:
            Design(measures=['same', 'same'], processes=[LocalTrend('test')])
        self.assertIn("Duplicates", cm.exception.args[0])

        with self.assertRaises(ValueError) as cm:
            Design(processes=[LocalTrend(id='same'), LocalTrend(id='same')], measures=['test'])
        self.assertEqual(cm.exception.args[0], "Duplicate process-ids: same.")

        with self.assertRaises(ValueError) as cm:
            Design(processes=[LocalTrend(id='1')], measures=['1'])
        self.assertIn("The following `measures` are not in any of the `processes`:\n{'1'}",
                      cm.exception.args[0])

    def test_design_f(self):
        # design
        design = simple_mv_velocity_design()
        batch_design = design.for_batch(num_groups=2, num_timesteps=1)

        # F doesn't require grad:
        self.assertFalse(batch_design.F(0).requires_grad)

    def test_design_q(self):
        # design
        design = simple_mv_velocity_design()
        batch_design = design.for_batch(num_groups=2, num_timesteps=1)

        # Q requires grad:
        self.assertTrue(batch_design.Q(0).requires_grad)

        # symmetric
        design_Q = batch_design.Q(0)[0].data.numpy()
        self.assertTrue(np.isclose(design_Q, design_Q.T).all(), msg="Covariance is not symmetric.")

    def test_design_h(self):
        # design
        design = simple_mv_velocity_design()
        batch_design = design.for_batch(num_groups=1, num_timesteps=1)

        design_H = batch_design.H(0)
        state_mean = Tensor([[[1.], [-.5],
                              [-1.5], [0.]]])
        measured_state = design_H.bmm(state_mean)
        self.assertListEqual(list1=measured_state.tolist(), list2=[[[1.0], [-1.5]]])

    def test_design_h_batch_process(self):
        #
        vel_1 = LocalTrend(id='vel_1')
        vel_1.add_measure('measure_1')

        vel_2 = LocalTrend(id='vel_2')
        vel_2.add_measure('measure_2')

        vel_common = LocalTrend(id='vel_common')
        vel_common.add_measure('measure_1')
        vel_common._set_measure('measure_1', 'position', 1., force=True)
        vel_common.add_measure('measure_2')
        vel_common._set_measure('measure_2', 'position', 5., force=True)

        design = Design(processes=[vel_1, vel_2, vel_common], measures=['measure_1', 'measure_2'])
        batch_design = design.for_batch(num_groups=1, num_timesteps=1)

        design_H = batch_design.H(0)

        self.assertListEqual(list1=design_H[0].tolist(),
                             list2=[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 5.0, 0.0]])

    def test_design_r(self):
        design = simple_mv_velocity_design(3)
        batch_design = design.for_batch(2, 1)

        cov = batch_design.R(0)[0]
        self.assertTupleEqual(cov.size(), (3, 3))

        self.assertTrue(cov.requires_grad)
        cholesky_log_diag = design.measure_covariance.param_dict()['cholesky_log_diag']
        cholesky_off_diag = design.measure_covariance.param_dict()['cholesky_off_diag']

        cov = cov.data.numpy()
        self.assertTrue(np.isclose(cov, cov.T).all(), msg="Covariance is not symmetric.")
        chol = cholesky(cov)

        for a, b in zip(torch.exp(cholesky_log_diag).tolist(), np.diag(chol).tolist()):
            self.assertAlmostEqual(a, b, places=4)

        for a, b in zip(cholesky_off_diag.tolist(), chol[np.tril_indices_from(chol, k=-1)].tolist()):
            self.assertAlmostEqual(a, b, places=4)
