import itertools
from unittest import TestCase

import torch
from parameterized import parameterized
from torch import nn

import numpy as np
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LinearModel, LocalLevel, NN


class TestProcess(TestCase):
    def test_decay(self):
        data = torch.zeros((2, 5, 1))
        kf = KalmanFilter(processes=[LocalLevel(id='lm', decay=(.95, 1.))], measures=['y'])
        # TODO

    def test_nn(self):
        y = torch.zeros((2, 5, 1))
        proc = NN(id='nn', nn=nn.Linear(in_features=10, out_features=2, bias=False))
        kf = KalmanFilter(processes=[proc], measures=['y'])
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
            kf(data, X=torch.zeros((num_groups, 5, wrong_dim)))
        expected = f"produced output with shape [{num_groups}, {wrong_dim}], but expected ({num_preds},) " \
                   f"or (num_groups, {num_preds}). Input had shape [{num_groups}, {wrong_dim}]"
        self.assertIn(expected, str(cm.exception))

        kf(data, X=torch.ones(num_groups, data.shape[1], num_preds))
