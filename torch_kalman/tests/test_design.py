import torch
from numpy import array_equal
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.design import Design
from torch_kalman.process.processes.local_trend import LocalTrend

from torch_kalman.tests import TestCaseTK, simple_mv_velocity_design


class TestDesign(TestCaseTK):
    def test_design_attrs(self):
        with self.assertRaises(AssertionError) as cm:
            Design(measures=['same', 'same'], processes=[])
        self.assertEqual(cm.exception.args[0], "Duplicate measures.")

        with self.assertRaises(ValueError) as cm:
            Design(processes=[LocalTrend(id='same'), LocalTrend(id='same')], measures=[])
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

        # we can't add batch-specific values that have already been set:
        vd_init = torch.randn(2)
        vel_damp = Parameter(vd_init)
        with self.assertRaises(ValueError) as cm:
            batch_design.processes['0'].set_transition(from_element='velocity', to_element='velocity', values=vel_damp)
        self.assertEqual(first=cm.exception.args[0],
                         second="The transition from 'velocity' to 'velocity' was already set for this Process, so can't "
                                "give it batch-specific values (unless set to `None`).")

    def test_design_q(self):
        # design
        design = simple_mv_velocity_design()
        batch_design = design.for_batch(num_groups=2, num_timesteps=1)

        # Q requires grad:
        self.assertTrue(batch_design.Q(0).requires_grad)

        # symmetric
        design_Q = batch_design.Q(0)[0].data.numpy()
        self.assertTrue(array_equal(design_Q, design_Q.T), msg="Covariance is not symmetric.")

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
        vel_common.add_measure('measure_1', value=None)
        vel_common.add_measure('measure_2', value=None)

        design = Design(processes=[vel_1, vel_2, vel_common], measures=['measure_1', 'measure_2'])

        # since it's None, requires batch-specific param:
        with self.assertRaises(ValueError) as cm:
            batch_design = design.for_batch(num_groups=2, num_timesteps=1)
        expected_msg_start = f"The measurement value for measure 'measure_1' of process 'vel_common' is None"
        self.assertIn(expected_msg_start, cm.exception.args[0])

        vel_common2 = LocalTrend(id='vel_common')
        vel_common2.add_measure('measure_1', value=lambda x: Tensor([1.0, 0.0]))
        vel_common2.add_measure('measure_2', value=lambda x: Tensor([0.0, 1.0]))
        design = Design(processes=[vel_1, vel_2, vel_common2], measures=['measure_1', 'measure_2'])
        batch_design = design.for_batch(num_groups=2, num_timesteps=1)

        design_H = batch_design.H(0)

        self.assertListEqual(list1=design_H[0].tolist(),
                             list2=[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

        self.assertListEqual(list1=design_H[1].tolist(),
                             list2=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])

    def test_design_r(self):
        design = simple_mv_velocity_design(3)
        batch_design = design.for_batch(2, 1)

        cov = batch_design.R(0)[0]
        self.assertTupleEqual(cov.size(), (3, 3))

        self.assertTrue(cov.requires_grad)

        self.check_covariance_chol(cov=cov,
                                   cholesky_log_diag=design.measure_cholesky_log_diag,
                                   cholesky_off_diag=design.measure_cholesky_off_diag)
