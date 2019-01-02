from typing import Union, Tuple, Sequence, Optional, TypeVar

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal as TorchMultivariateNormal, Distribution

from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief

import numpy as np

from torch_kalman.state_belief.over_time import StateBeliefOverTime


# noinspection PyPep8Naming
class Gaussian(StateBelief):
    def predict(self, F: Tensor, Q: Tensor) -> StateBelief:
        Ft = F.permute(0, 2, 1)
        means = torch.bmm(F, self.means[:, :, None]).squeeze(2)
        covs = torch.bmm(torch.bmm(F, self.covs), Ft) + Q
        return self.__class__(means=means, covs=covs, last_measured=self.last_measured + 1)

    def update(self, obs: Tensor) -> StateBelief:
        assert isinstance(obs, Tensor)
        measured_means, system_covariance = self.measurement
        residuals = obs - measured_means
        K = self.kalman_gain(system_covariance)

        # clone tensors since autograd can't handle in-place changes
        means_new = self.means.clone()
        covs_new = self.covs.clone()

        # handle kalman-update for groups w/missing values:
        isnan = (obs != obs)
        anynan_by_group = (torch.sum(isnan, 1) > 0)
        nan_group_idx = anynan_by_group.nonzero().squeeze(-1)
        for i in nan_group_idx:
            group_isnan = isnan[i]
            if group_isnan.all():  # if all nan, just don't perform update
                continue
            # if partial nan, perform partial update:
            raise NotImplementedError("Partial update not currently implemented; please report error to package maintainer")

        # faster kalman-update for groups w/o missing values
        nonan_g = (~anynan_by_group).nonzero().squeeze(-1)
        if len(nonan_g) > 0:
            means_new[nonan_g] = self.means[nonan_g] + torch.bmm(K[nonan_g], residuals[nonan_g].unsqueeze(2)).squeeze(2)
            covs_new[nonan_g] = self.covariance_update(self.covs[nonan_g], K[nonan_g], self.H[nonan_g], self.R[nonan_g])

        any_measured_group_idx = (torch.sum(~isnan, 1) > 0).nonzero().squeeze(-1)
        last_measured = self.last_measured.clone()
        last_measured[any_measured_group_idx] = 0
        return self.__class__(means=means_new, covs=covs_new, last_measured=last_measured)

    def kalman_gain(self, system_covariance, method='solve'):
        Ht = self.H.permute(0, 2, 1)
        covs_measured = torch.bmm(self.covs, Ht)
        if method == 'solve':
            A = system_covariance.permute(0, 2, 1)
            B = covs_measured.permute(0, 2, 1)
            Kt, _ = torch.gesv(B, A)
            K = Kt.permute(0, 2, 1)
        elif method == 'inverse':
            Sinv = torch.cat([torch.inverse(system_covariance[i, :, :]).unsqueeze(0)
                              for i in range(len(system_covariance))], 0)
            K = torch.bmm(covs_measured, Sinv)
        else:
            raise ValueError(f"Unrecognized method '{method}'.")
        return K

    @staticmethod
    def covariance_update(covariance: Tensor, K: Tensor, H: Tensor, R: Tensor) -> Tensor:
        """
        "Joseph stabilized" covariance correction.
        """
        rank = covariance.shape[1]
        I = torch.eye(rank, rank, device=covariance.device).expand(len(covariance), -1, -1)
        p1 = (I - torch.bmm(K, H))
        p2 = torch.bmm(torch.bmm(p1, covariance), p1.permute(0, 2, 1))
        p3 = torch.bmm(torch.bmm(K, R), K.permute(0, 2, 1))
        return p2 + p3

    @classmethod
    def concatenate_over_time(cls,
                              state_beliefs: Sequence['Gaussian'],
                              design: Design,
                              start_datetimes: Optional[np.ndarray] = None) -> 'GaussianOverTime':
        return GaussianOverTime(state_beliefs=state_beliefs, design=design, start_datetimes=start_datetimes)

    def to_distribution(self) -> Distribution:
        return MultivariateNormal(loc=self.means, covariance_matrix=self.covs)


class GaussianOverTime(StateBeliefOverTime):
    @property
    def distribution(self) -> TypeVar('Distribution'):
        return MultivariateNormal


class MultivariateNormal(TorchMultivariateNormal):
    """
    Workaround for https://github.com/pytorch/pytorch/issues/11333
    """

    def __init__(self, loc: Tensor, covariance_matrix: Tensor, validate_args: bool = False):
        super().__init__(loc=loc, covariance_matrix=covariance_matrix, validate_args=validate_args)
        self.univariate = len(self.event_shape) == 1 and self.event_shape[0] == 1

    def log_prob(self, value):
        if self.univariate:
            value = torch.squeeze(value, -1)
            mean = torch.squeeze(self.loc, -1)
            var = torch.squeeze(torch.squeeze(self.covariance_matrix, -1), -1)
            numer = -torch.pow(value - mean, 2) / (2. * var)
            denom = .5 * torch.log(2. * np.pi * var)
            log_prob = numer - denom
        else:
            log_prob = super().log_prob(value)
        return log_prob

    def rsample(self, sample_shape=None):
        if self.univariate:
            std = torch.sqrt(torch.squeeze(self.covariance_matrix, -1))
            eps = self.loc.new(*self._extended_shape(sample_shape)).normal_()
            return std * eps + self.loc
        else:
            return super().rsample(sample_shape=sample_shape)
