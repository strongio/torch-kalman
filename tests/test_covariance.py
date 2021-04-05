import torch
from torchcast.covariance import Covariance
import unittest


class TestCovariance(unittest.TestCase):
    torch.no_grad()

    def test_from_log_cholesky(self):
        module = Covariance(id='test', rank=3)

        module.state_dict()['cholesky_log_diag'][:] = torch.arange(1., 3.1)
        module.state_dict()['cholesky_off_diag'][:] = torch.arange(1., 3.1)

        expected = torch.tensor([[7.3891, 2.7183, 5.4366],
                                 [2.7183, 55.5982, 24.1672],
                                 [5.4366, 24.1672, 416.4288]])
        diff = (expected - module(None)).abs()
        self.assertTrue((diff < .0001).all())

    def test_empty_idx(self):
        module = torch.jit.script(Covariance(id='test', rank=3, empty_idx=[0]))
        cov = module(None)
        self.assertTrue((cov[0, :] == 0).all())
        self.assertTrue((cov[:, 0] == 0).all())
        self.assertTrue((cov == cov.t()).all())
