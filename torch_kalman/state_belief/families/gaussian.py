from collections import defaultdict
from typing import Sequence, Optional, TypeVar

import torch

from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief

import numpy as np

from torch_kalman.state_belief.distributions.multivariate_normal import MultivariateNormal
from torch_kalman.state_belief.over_time import StateBeliefOverTime

from torch_kalman.state_belief.utils import bmat_idx


# noinspection PyPep8Naming
class Gaussian(StateBelief):
    distribution = MultivariateNormal

    def __init__(self, means: Tensor, covs: Tensor, last_measured: Optional[Tensor] = None):
        super().__init__(means=means, covs=covs, last_measured=last_measured)

    def predict(self, F: Tensor, Q: Tensor) -> StateBelief:
        Ft = F.permute(0, 2, 1)
        means = torch.bmm(F, self.means[:, :, None]).squeeze(2)
        covs = torch.bmm(torch.bmm(F, self.covs), Ft) + Q
        return self.__class__(means=means, covs=covs, last_measured=self.last_measured + 1)

    def update(self, obs: Tensor) -> StateBelief:
        assert isinstance(obs, Tensor)
        isnan = (obs != obs)
        num_dim = obs.shape[1]

        means_new = self.means.data.clone()
        covs_new = self.covs.data.clone()

        # need to do a different update depending on which (if any) dimensions are missing:
        update_groups = defaultdict(list)
        anynan_by_group = (torch.sum(isnan, 1) > 0)

        # groups with nan:
        nan_group_idx = anynan_by_group.nonzero().squeeze(-1).tolist()
        for i in nan_group_idx:
            if isnan[i].all():
                continue  # if all nan, then simply skip update
            which_valid = (~isnan[i]).nonzero().squeeze(-1).tolist()
            update_groups[tuple(which_valid)].append(i)

        update_groups = list(update_groups.items())

        # groups without nan:
        if isnan.any():
            nonan_group_idx = (~anynan_by_group).nonzero().squeeze(-1).tolist()
            if len(nonan_group_idx):
                update_groups.append((slice(None), nonan_group_idx))
        else:
            # if no nans at all, then faster to use slices:
            update_groups.append((slice(None), slice(None)))

        measured_means, system_covs = self.measurement

        # updates:
        for which_valid, group_idx in update_groups:
            idx_2d = bmat_idx(group_idx, which_valid)
            idx_3d = bmat_idx(group_idx, which_valid, which_valid)
            group_obs = obs[idx_2d]
            group_means = self.means[group_idx]
            group_covs = self.covs[group_idx]
            group_measured_means = measured_means[idx_2d]
            group_system_covs = system_covs[idx_3d]
            group_H = self.H[idx_2d]
            group_R = self.R[idx_3d]
            group_K = self.kalman_gain(system_covariance=group_system_covs, covariance=group_covs, H=group_H)
            means_new[group_idx] = self.mean_update(means=group_means, K=group_K, residuals=group_obs - group_measured_means)
            covs_new[group_idx] = self.covariance_update(covariance=group_covs, K=group_K, H=group_H, R=group_R)

        # calculate last-measured:
        any_measured_group_idx = (torch.sum(~isnan, 1) > 0).nonzero().squeeze(-1)
        last_measured = self.last_measured.clone()
        last_measured[any_measured_group_idx] = 0
        return self.__class__(means=means_new, covs=covs_new, last_measured=last_measured)

    @staticmethod
    def mean_update(means: Tensor, K: Tensor, residuals: Tensor):
        return means + K.matmul(residuals.unsqueeze(2)).squeeze(2)

    @staticmethod
    def covariance_update(covariance: Tensor, K: Tensor, H: Tensor, R: Tensor) -> Tensor:
        """
        "Joseph stabilized" covariance correction.
        """
        rank = covariance.shape[1]
        I = torch.eye(rank, rank, device=covariance.device).expand(len(covariance), -1, -1)
        p1 = I - K.matmul(H)
        p2 = p1.matmul(covariance).matmul(p1.permute(0, 2, 1))  # p2=torch.bmm(torch.bmm(p1,covariance),p1.permute(0, 2, 1))
        p3 = K.matmul(R).matmul(K.permute(0, 2, 1))  # p3 = torch.bmm(torch.bmm(K, R), K.permute(0, 2, 1))
        return p2 + p3

    @staticmethod
    def kalman_gain(covariance: Tensor,
                    system_covariance: Tensor,
                    H: Tensor,
                    method: str = 'solve'):

        Ht = H.permute(0, 2, 1)
        covs_measured = torch.bmm(covariance, Ht)
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

    @classmethod
    def concatenate_over_time(cls, state_beliefs: Sequence['Gaussian'], design: Design) -> 'GaussianOverTime':
        return GaussianOverTime(state_beliefs=state_beliefs, design=design)

    def sample(self, eps: Optional[Tensor] = None) -> Tensor:
        torch_distribution = self.distribution(self.means, self.covs)
        return torch_distribution.deterministic_sample(eps=eps)

    def log_prob(self, obs: Tensor) -> Tensor:
        measured_means, system_covariance = self.measurement
        dist = self.distribution(measured_means, system_covariance)
        return dist.log_prob_with_missings(obs=obs)


class GaussianOverTime(StateBeliefOverTime):
    @property
    def distribution(self) -> TypeVar('MultivariateNormal'):
        return self.family.distribution
