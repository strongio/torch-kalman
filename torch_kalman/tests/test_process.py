import torch
from numpy import datetime64
from torch import Tensor

from torch_kalman.process.season import Season
from torch_kalman.process.velocity import Velocity

from torch_kalman.tests import TestCaseTK


class TestProcess(TestCaseTK):
    def test_velocity_covariance(self):
        vel = Velocity(id='test')
        batch_vel = vel.for_batch(batch_size=1)

        self.check_covariance_chol(cov=batch_vel.Q()[0],
                                   cholesky_log_diag=vel.cholesky_log_diag,
                                   cholesky_off_diag=vel.cholesky_off_diag)

    def test_velocity_transition(self):
        batch_vel = Velocity(id='test').for_batch(batch_size=1)

        # check F:
        self.assertListEqual(list1=batch_vel.F()[0].tolist(), list2=[[1., 1.], [0., 1.]])
        state_mean = Tensor([[1.], [-.5]])
        for i in range(3):
            state_mean = torch.mm(batch_vel.F()[0], state_mean)
            self.assertEqual(state_mean[0].item(), 1 - .5 * (i + 1.))
            self.assertEqual(state_mean[1].item(), -.5)

    def test_seasons(self):
        # test seasons without durations
        season = Season(id='day_of_week', num_seasons=7, season_duration=1, start_datetime=datetime64('2018-01-01'))

        # need to include start_datetimes since included above
        with self.assertRaises(ValueError) as cm:
            s1_batch = season.for_batch(batch_size=1, time=0)
        self.assertEqual(cm.exception.args[0], "`start_datetimes` argument required.")

        # TODO: test F
        # TODO: test H
        # TODO: test Q

    def test_seasons_with_durations(self):
        season = Season(id='week_of_year', num_seasons=52, season_duration=7, start_datetime=datetime64('2018-12-31'))

        # TODO: test F
        # TODO: test start_datetime
