from unittest import TestCase

import numpy as np
import torch

from torch_kalman.kalman_filters.forecast import Forecast


class TestBasicSeasons(TestCase):
    def test_weekly_cycle(self):
        torch.manual_seed(32)
        mse, final_loss, final_iter = self._test_seasons(horizon=1, offset=0, duration=1, period=7)
        self.assertEqual(final_iter, 35)
        self.assertAlmostEqual(final_loss, 1.3680, places=2)
        self.assertAlmostEqual(mse, 0.9036, places=2)

    def test_season_with_duration(self):
        # a season with duration:
        torch.manual_seed(32)
        mse, final_loss, final_iter = self._test_seasons(horizon=1, offset=0, duration=7, period=5)
        self.assertEqual(final_iter, 40)
        self.assertAlmostEqual(final_loss, 1.497, places=2)
        self.assertAlmostEqual(mse, 1.169, places=2)

    @staticmethod
    def make_simple_series(duration, period):
        x = []
        for _ in range(4):
            for i in range(period):
                for _ in range(duration):
                    x.append(i)
        x = np.array(x, dtype='float')
        return x

    @staticmethod
    def make_simple_model(horizon, period, duration, offset):
        model = Forecast(measures=['0'], horizon=horizon)
        model.add_level('0')
        model.add_season('0', period=period, duration=duration, season_start=None if duration == 1 else offset)
        model.finalize()
        return model

    @staticmethod
    def series1d_to_kf_input(series):
        kf_input = torch.autograd.Variable(torch.from_numpy(series).float())
        kf_input = kf_input[None, :, None]
        return kf_input

    @staticmethod
    def train_model(kf_input, time_start, model, max_iter=150):
        assert max_iter > 0
        opt = torch.optim.Adam(model.parameters(), lr=.10)
        for t in range(max_iter):
            log_lik = model.log_likelihood(kf_input, time_start=time_start)
            loss = -torch.mean(log_lik[log_lik == log_lik])
            opt.zero_grad()
            loss.backward()
            opt.step()
            # noinspection PyUnboundLocalVariable
            if t > 0 and torch.abs(prev_loss - loss).data.tolist() < .001:
                break
            prev_loss = loss

        # noinspection PyUnboundLocalVariable
        return loss, t

    def _test_seasons(self, horizon, offset, duration, period):
        series = self.make_simple_series(duration, period)

        # jitter:
        np.random.seed(32)
        series += np.random.normal(size=series.size)

        # model:
        model = self.make_simple_model(horizon, period, duration, offset)

        # inputs:
        time_start = torch.autograd.Variable(torch.Tensor([offset]))
        kf_input = self.series1d_to_kf_input(series)

        # train:
        final_loss, final_iter = self.train_model(kf_input, time_start, model)

        # mse:
        predicted = model(kf_input, time_start=time_start)
        mse = np.nanmean(np.power((kf_input - predicted).data.numpy(), 2))

        return mse, final_loss.data.tolist(), final_iter
