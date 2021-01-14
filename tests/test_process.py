import itertools
from unittest import TestCase

import torch
from parameterized import parameterized
from torch import Tensor

import numpy as np
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LinearModel, LocalLevel


class TestProcess(TestCase):
    def test_decay(self):
        data = torch.zeros((2, 5, 1))
        kf = KalmanFilter(processes=[LocalLevel(id='lm', decay=(.95, 1.))], measures=['y'])
        # TODO

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2, 3]))
    @torch.no_grad()
    def test_lm(self, num_groups: int = 1, num_preds: int = 1):
        data = torch.zeros((num_groups, 5, 1))
        kf = KalmanFilter(
            processes=[
                LinearModel(id='lm', predictors=[f"x{i}" for i in range(num_preds)])
            ],
            measures=['y']
        )
        wrong_dim = 1 if num_preds > 1 else 2
        with self.assertRaises((RuntimeError, torch.jit.Error), msg=(num_groups, num_preds)) as cm:
            kf(data, predictors=torch.zeros((num_groups, 5, wrong_dim)))
        expected = f"produced output with shape [{num_groups}, {wrong_dim}], but expected ({num_preds},) " \
                   f"or (num_groups, {num_preds}). Input had shape [{num_groups}, {wrong_dim}]"
        self.assertIn(expected, str(cm.exception))

        kf(data, predictors=torch.ones(num_groups, data.shape[1], num_preds))
