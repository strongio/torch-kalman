import torch
from torch import Tensor
from torch.distributions import MultivariateNormal as TorchMultivariateNormal
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal

from math import pi


class MultivariateNormal(TorchMultivariateNormal):
    def __init__(self, loc: Tensor, covariance_matrix: Tensor, validate_args: bool = False):
        super().__init__(loc=loc, covariance_matrix=covariance_matrix, validate_args=validate_args)
        self.univariate = len(self.event_shape) == 1 and self.event_shape[0] == 1

    def log_prob(self, value: Tensor) -> Tensor:
        if self.univariate:
            value = torch.squeeze(value, -1)
            mean = torch.squeeze(self.loc, -1)
            var = torch.squeeze(torch.squeeze(self.covariance_matrix, -1), -1)
            numer = -torch.pow(value - mean, 2) / (2. * var)
            denom = .5 * torch.log(2. * pi * var)
            log_prob = numer - denom
        else:
            log_prob = super().log_prob(value)
        return log_prob

    def deterministic_sample(self, sample_shape=None, eps=None):
        if self.univariate:
            if eps is None:
                eps = self.loc.new(*self._extended_shape(sample_shape)).normal_()
            std = torch.sqrt(torch.squeeze(self.covariance_matrix, -1))
            return std * eps + self.loc
        else:
            if eps is None:
                shape = self._extended_shape(sample_shape)
                eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
            return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError
