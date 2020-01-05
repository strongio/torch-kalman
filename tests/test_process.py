from unittest import TestCase

import torch
from torch import Tensor

import numpy as np

from torch_kalman.design import Design
from torch_kalman.process import Season, FourierSeason, Process
from torch_kalman.process.processes.local_trend import LocalTrend
from torch_kalman.process.processes.season.fourier import TBATS


class TestProcess(TestCase):

    def test_fourier_season(self):
        season = FourierSeason(id='season', seasonal_period=24, K=2, decay=False, dt_unit=None)
        season.add_measure('measure')
        design = Design(processes=[season], measures=['measure'])
        for_batch = design.for_batch(1, 24 * 2)

        positions = []
        state = torch.randn(5)
        for i in range(for_batch.num_timesteps):
            state = for_batch.F(i)[0].matmul(state)
            positions.append(round(state[-1].item() * 100) / 100.)

        self.assertListEqual(positions[0:24], positions[-24:])

    def test_tbats_season(self):
        K = 3
        season = TBATS(id='season', seasonal_period=24, K=K, decay=False, dt_unit=None)
        season.add_measure('measure')
        design = Design(processes=[season], measures=['measure'])
        for_batch = design.for_batch(1, 24 * 7)

        positions = []
        state = torch.randn(int(K * 2))
        for i in range(for_batch.num_timesteps):
            state = for_batch.F(i)[0].matmul(state)
            pos = for_batch.H(i)[0].matmul(state)
            positions.append(round(pos.item() * 100) / 100.)

        self.assertListEqual(positions[0:24], positions[-24:])
        # import matplotlib
        # matplotlib.use('TkAgg')
        # from matplotlib import pyplot as plt
        # plt.plot(positions)
        # plt.show()

    def test_velocity(self):
        # no decay:
        lt = LocalTrend(id='test', decay_velocity=False)
        lt.add_measure('measure')
        design = Design(processes=[lt], measures=['measure'])
        batch_vel = design.for_batch(2, 1)

        # check F:
        self.assertListEqual(list1=batch_vel.F(0)[0].tolist(), list2=[[1., 1.], [0., 1.]])
        state_mean = Tensor([[1.], [-.5]])
        for i in range(3):
            state_mean = torch.mm(batch_vel.F(0)[0], state_mean)
            self.assertEqual(state_mean[0].item(), 1 - .5 * (i + 1.))
            self.assertEqual(state_mean[1].item(), -.5)

        # with decay:
        lt = LocalTrend(id='test', decay_velocity=(.50, 1.00))
        lt.add_measure('measure')
        design = Design(processes=[lt], measures=['measure'])
        batch_vel = design.for_batch(2, 1)
        self.assertLess(batch_vel.F(0)[0][1, 1], 1.0)
        self.assertGreater(batch_vel.F(0)[0][1, 1], 0.5)
        decay = design.processes['test'].decayed_transitions['velocity'].get_value()
        self.assertEqual(decay, batch_vel.F(0)[0][1, 1])

        state_mean = Tensor([[0.], [1.0]])
        for i in range(3):
            state_mean = torch.mm(batch_vel.F(0)[0], state_mean)
            self.assertEqual(decay ** (i + 1), state_mean[1].item())

    def test_discrete_seasons(self):
        # test seasons without durations
        season = Season(id='day_of_week', seasonal_period=7, season_duration=1, season_start='2018-01-01',
                        dt_unit='D')
        season.add_measure('measure')

        # need to include start_datetimes since included above
        with self.assertRaises(ValueError) as cm:
            season.for_batch(1, 1)
        self.assertIn('Must pass `start_datetimes`', cm.exception.args[0])

        design = Design(processes=[season], measures=['measure'])
        batch_season = design.for_batch(1, 1, start_datetimes=np.array([np.datetime64('2018-01-01')]))

        # test transitions manually:
        state_mean = torch.arange(0.0, 7.0)[:, None]
        state_mean[0] = -state_mean[1:].sum()
        for i in range(10):
            state_mean_last = state_mean
        state_mean = torch.mm(batch_season.F(0)[0], state_mean)
        self.assertTrue((state_mean[1:] == state_mean_last[:-1]).all())

        self.assertListEqual(batch_season.H(0)[0].tolist(), [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def test_cls_validation(self):
        with self.assertRaises(TypeError):
            class MyProcess(Process):
                def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'Process':
                    return super().for_batch(num_groups, num_timesteps)

        with self.assertRaises(TypeError):
            class MyProcess(Process):
                def for_batch(self, num_groups, num_timesteps, pred_mat=None) -> 'Process':
                    return super().for_batch(num_groups, num_timesteps)

                def initial_state_means_for_batch(self, parameters, num_groups) -> Tensor:
                    return super().initial_state_means_for_batch(parameters, num_groups)
