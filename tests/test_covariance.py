import torch
from torch_kalman.covariance import Covariance
import unittest


class TestCovariance(unittest.TestCase):
    def test_from_log_cholesky(self):
        covs = Covariance.from_log_cholesky(log_diag=torch.arange(1., 3.1).expand(3, -1),
                                            off_diag=torch.arange(1., 3.1).expand(3, -1))

        gt = torch.tensor([[7.3891, 2.7183, 5.4366],
                           [2.7183, 55.5982, 24.1672],
                           [5.4366, 24.1672, 416.4288]])
        for cov in covs:
            diff = (gt - cov).abs()
            self.assertTrue((diff < .0001).all())
