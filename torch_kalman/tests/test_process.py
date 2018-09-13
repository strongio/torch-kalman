import torch
from torch import Tensor

from torch_kalman.process.velocity import Velocity

from torch_kalman.tests import TestCaseTK


class TestProcess(TestCaseTK):
    def test_velocity_covariance(self):
        vel = Velocity(id='test')
        batch_vel = vel.for_batch(batch_size=1)

        self.check_covariance_chol(cov=batch_vel.Q[0],
                                   cholesky_log_diag=vel.cholesky_log_diag,
                                   cholesky_off_diag=vel.cholesky_off_diag)

    def test_velocity_transition(self):
        batch_vel = Velocity(id='test').for_batch(batch_size=1)

        # check F:
        self.assertListEqual(list1=batch_vel.F[0].tolist(), list2=[[1., 1.], [0., 1.]])
        state_mean = Tensor([[1.], [-.5]])
        for i in range(3):
            state_mean = torch.mm(batch_vel.F[0], state_mean)
            self.assertEqual(state_mean[0].item(), 1 - .5 * (i + 1.))
            self.assertEqual(state_mean[1].item(), -.5)
