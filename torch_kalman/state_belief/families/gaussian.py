from typing import Union, Tuple, Sequence, Optional
from warnings import warn

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
        return self.__class__(means=means, covs=covs)

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
        nan_groups = anynan_by_group.nonzero().squeeze(-1)
        for i in nan_groups:
            group_isnan = isnan[i]
            if group_isnan.all():  # if all nan, just don't perform update
                continue
            # if partial nan, perform partial update:
            means_new[i], covs_new[i] = self.partial_update(valid_idx=(~group_isnan).nonzero().squeeze(1),
                                                            mean=self.means[i], cov=self.covs[i],
                                                            residual=residuals[i], K=K[i], H=self.H[i], R=self.R[i])

        # faster kalman-update for groups w/o missing values
        nonan_g = (~anynan_by_group).nonzero().squeeze(-1)
        if len(nonan_g) > 0:
            means_new[nonan_g] = self.means[nonan_g] + torch.bmm(K[nonan_g], residuals[nonan_g].unsqueeze(2)).squeeze(2)
            covs_new[nonan_g] = self.covariance_update(self.covs[nonan_g], K[nonan_g], self.H[nonan_g], self.R[nonan_g])

        return self.__class__(means=means_new, covs=covs_new)

    def partial_update(self,
                       valid_idx: Union[Tensor, Sequence[int], np.ndarray],
                       mean: Tensor,
                       cov: Tensor,
                       residual: Tensor,
                       K: Tensor,
                       H: Tensor,
                       R: Tensor) -> Tuple[Tensor, Tensor]:

        residual = residual[valid_idx]
        K = K[:, valid_idx]
        H = H[valid_idx, :]
        R = R[valid_idx][:, valid_idx]
        mean = mean + torch.mm(K, residual.unsqueeze(1)).squeeze(1)
        cov = self.covariance_update(covariance=cov[None, :, :], K=K[None, :, :], H=H[None, :, :], R=R[None, :, :])[0]
        return mean, cov

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
                              design: Optional[Design] = None) -> 'GaussianOverTime':
        return GaussianOverTime(state_beliefs=state_beliefs, design=design)

    def to_distribution(self) -> Distribution:
        return MultivariateNormal(loc=self.means, covariance_matrix=self.covs)


class GaussianOverTime(StateBeliefOverTime):
    @property
    def distribution(self):
        return MultivariateNormal


class MultivariateNormal(TorchMultivariateNormal):
    """
    Workaround for https://github.com/pytorch/pytorch/issues/11333
    """

    def __init__(self, loc: Tensor, covariance_matrix: Tensor, validate_args: bool = False):
        assert loc.dim() == 3
        if loc.shape[2] == 1:
            self.univariate = True
            self.loc = loc
            self.covariance_matrix = covariance_matrix
        else:
            self.univariate = False
            if loc.device != torch.device('cpu'):
                warn("`MultivariateNormal` not recommended for gpu, consider moving Tensors to cpu. "
                     "See https://github.com/pytorch/pytorch/issues/11333")
            super().__init__(loc=loc, covariance_matrix=covariance_matrix, validate_args=validate_args)

    def log_prob(self, value):
        if self.univariate:
            assert value.shape[2] == 1
            value = torch.squeeze(value, 2)
            mean = torch.squeeze(self.loc, 2)
            var = self.covariance_matrix[:, :, 0, 0]
            numer = -torch.pow(value - mean, 2) / (2. * var)
            denom = .5 * torch.log(2. * np.pi * var)
            log_prob = numer - denom
        else:
            log_prob = super().log_prob(value)
        return log_prob

    def rsample(self, sample_shape):
        raise NotImplementedError("TODO")
        # if self.univariate:
        #     return torch.sqrt(self.covariance_matrix) * torch.randn(sample_shape) + self.loc
        # else:
        #     return super().rsample(sample_shape=sample_shape)
