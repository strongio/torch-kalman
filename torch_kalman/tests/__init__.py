import unittest

from numpy import array_equal, diag, tril_indices_from
from numpy.linalg import cholesky

import torch
from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.measure import Measure
from torch_kalman.process.velocity import Velocity

import numpy as np


def simple_mv_velocity_design(dims=2):
    processes, measures = [], []
    for i in range(dims):
        process = Velocity(id=str(i))
        measure = Measure(id=str(i))
        measure.add_process(process, value=1.)
        processes.append(process)
        measures.append(measure)
    return Design(processes=processes, measures=measures)


def make_rw_data(num_timesteps=100, num_groups=5, measure_cov=((1., .5), (.5, 1.))):
    num_dims = len(measure_cov)
    velocity = 1 / np.tile(np.arange(1, num_timesteps + 1), (num_groups, 1))
    velocity = np.stack([velocity] * num_dims, axis=2)
    state_means = np.cumsum(velocity, axis=1)
    white_noise = np.random.multivariate_normal(size=(num_groups, num_timesteps),
                                                mean=np.zeros(num_dims),
                                                cov=measure_cov)
    return Tensor(state_means + white_noise)


class TestCaseTK(unittest.TestCase):
    def check_covariance_chol(self, cov: Tensor, cholesky_log_diag: Tensor, cholesky_off_diag: Tensor):
        cov = cov.data.numpy()
        self.assertTrue(array_equal(cov, cov.T), msg="Covariance is not symmetric.")
        chol = cholesky(cov)

        for a, b in zip(torch.exp(cholesky_log_diag).tolist(), diag(chol).tolist()):
            self.assertAlmostEqual(a, b, places=4,
                                   msg=f"Couldn't verify the log-diagonal of the cholesky factorization of covariance:{cov}")

        for a, b in zip(cholesky_off_diag.tolist(), chol[tril_indices_from(chol, k=-1)].tolist()):
            self.assertAlmostEqual(a, b, places=4,
                                   msg=f"Couldn't verify the off-diagonal of the cholesky factorization of covariance:{cov}")
