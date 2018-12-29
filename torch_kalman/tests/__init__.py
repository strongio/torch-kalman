import unittest

from numpy import array_equal, diag, tril_indices_from
from numpy.linalg import cholesky

import torch
from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.process import Process, FourierSeason, LocalLevel, Season, LinearModel
from torch_kalman.process.processes.local_trend import LocalTrend

import numpy as np

from torch_kalman.process.processes.nn import NN


def simple_mv_velocity_design(dims=2):
    processes, measures = [], []
    for i in range(dims):
        process = LocalTrend(id=str(i), decay_velocity=False)
        measure = str(i)
        process.add_measure(measure=measure)
        processes.append(process)
        measures.append(measure)
    return Design(processes=processes, measures=measures)


def make_rw_data(num_timesteps=100, num_groups=5, measure_cov=((1., .5), (.5, 1.))):
    num_dims = len(measure_cov)
    velocity = 1 / np.tile(np.arange(1, num_timesteps + 1), (num_groups, 1))
    velocity = np.stack([velocity] * num_dims, axis=2)
    velocity[:, 0:10, :] -= 1.
    state_means = np.cumsum(velocity, axis=1)
    white_noise = np.random.multivariate_normal(size=(num_groups, num_timesteps),
                                                mean=np.zeros(num_dims),
                                                cov=measure_cov)
    return Tensor(state_means + white_noise)


def name_to_proc(id: str, **kwargs) -> Process:
    season_start = '2010-01-04'

    if 'hour_in_day' in id:
        out = FourierSeason(id=id,
                            seasonal_period=24, season_start=season_start, dt_unit='h',
                            **kwargs)
    elif 'day_in_year' in id:
        out = FourierSeason(id=id,
                            seasonal_period=24 * 364.25, season_start=season_start, dt_unit='h',
                            **kwargs)
    elif 'local_level' in id:
        out = LocalLevel(id=id, **kwargs)
    elif 'local_trend' in id:
        out = LocalTrend(id=id, **kwargs)
    elif 'day_in_week' in id:
        out = Season(id=id,
                     seasonal_period=7, season_duration=24,
                     season_start=season_start, dt_unit='h',
                     **kwargs)
    elif 'nn_predictors' in id:
        out = NN(id=id,
                 add_module_params_to_process=False,  # so we can use a separate parameter group
                 model_mat_kwarg_name='predictors',
                 **kwargs)
    elif 'predictors' in id:
        out = LinearModel(id=id,
                          covariates=self.predictors,
                          model_mat_kwarg_name='predictors',
                          **kwargs)
    else:
        raise NotImplementedError(f"Unsure what process to use for `{id}`.")

    return out


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
