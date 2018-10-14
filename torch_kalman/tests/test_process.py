import torch
from numpy import datetime64, array
from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.process.processes.season import Season
from torch_kalman.process.processes.velocity import Velocity

from torch_kalman.tests import TestCaseTK


class TestProcess(TestCaseTK):

    def test_velocity_transition(self):
        batch_vel = Velocity(id='test', dampened=False).for_batch(batch_size=1)

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
            season.for_batch(batch_size=1, time=0)
        self.assertEqual(cm.exception.args[0], "`start_datetimes` argument required.")

        batch_season = season.for_batch(batch_size=1, time=0, start_datetimes=array([datetime64('2018-01-01')]))

        # test transitions manually:
        state_mean = torch.arange(0.0, 7.0)[:, None]
        state_mean[0] = -state_mean[1:].sum()
        for i in range(10):
            state_mean_last = state_mean
            state_mean = torch.mm(batch_season.F()[0], state_mean)
            self.assertTrue((state_mean[1:] == state_mean_last[:-1]).all())

        # TODO: test H
        measure = 'measure'
        season.add_measure(measure)
        design = Design(processes=[season], measures=[measure])

        # TODO: test Q

    def test_seasons_with_durations(self):
        season = Season(id='week_of_year', num_seasons=52, season_duration=7, start_datetime=datetime64('2018-12-31'))

        # TODO: test F
        # TODO: test start_datetime
        # state_mean = torch.arange(0, 7)[None, :, None].expand(2, -1, -1)
        # self.assertTrue((state_mean[0] == state_mean[1]).all())
        # state_mean[:, 0, :] = -state_mean[:, 1:, :].sum(1)
        # for t in range(10):
        #     design_for_batch = design.for_batch(batch_size=2, time=t,
        #                                         start_datetimes=array([datetime64('2018-01-01'), datetime64('2018-01-02')]))
        #     state_mean_last = state_mean
        #     state_mean = torch.bmm(design_for_batch.F(), state_mean)
