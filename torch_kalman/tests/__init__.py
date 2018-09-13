import unittest

from numpy import array_equal, diag, tril_indices_from
from numpy.linalg import cholesky

import torch
from torch import Tensor


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
