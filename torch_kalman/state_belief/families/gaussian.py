from typing import Sequence, Optional, Union, Tuple

import torch

from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief

from torch_kalman.state_belief.over_time import StateBeliefOverTime, Selector

from torch_kalman.state_belief.utils import bmat_idx, deterministic_sample_mvnorm


# noinspection PyPep8Naming
class Gaussian(StateBelief):
    def __init__(self, means: Tensor, covs: Tensor, last_measured: Optional[Tensor] = None):
        self._measured_means = None
        self._system_uncertainty = None
        super().__init__(means=means, covs=covs, last_measured=last_measured)

    def _update_group(self,
                      obs: Tensor,
                      group_idx: Union[slice, Sequence[int]],
                      which_valid: Union[slice, Sequence[int]]) -> Tuple[Tensor, Tensor]:

        idx_2d = bmat_idx(group_idx, which_valid)
        idx_3d = bmat_idx(group_idx, which_valid, which_valid)
        group_obs = obs[idx_2d]
        group_means = self.means[group_idx]
        group_covs = self.covs[group_idx]
        group_measured_means = self.measured_means[idx_2d]
        group_system_covs = self.system_uncertainty[idx_3d]
        group_H = self.H[idx_2d]
        group_R = self.R[idx_3d]
        group_K = self.kalman_gain(system_covariance=group_system_covs, covariance=group_covs, H=group_H)
        means_new = self.mean_update(mean=group_means, K=group_K, residuals=group_obs - group_measured_means)
        covs_new = self.covariance_update(covariance=group_covs, K=group_K, H=group_H, R=group_R)
        return means_new, covs_new

    @staticmethod
    def covariance_update(covariance: Tensor, K: Tensor, H: Tensor, R: Tensor) -> Tensor:
        """
        "Joseph stabilized" covariance correction.
        """
        rank = covariance.shape[1]
        I = torch.eye(rank, rank, device=covariance.device).expand(len(covariance), -1, -1)
        p1 = I - K.matmul(H)
        p2 = p1.matmul(covariance).matmul(p1.permute(0, 2, 1))
        p3 = K.matmul(R).matmul(K.permute(0, 2, 1))
        return p2 + p3

    @staticmethod
    def kalman_gain(covariance: Tensor, system_covariance: Tensor, H: Tensor):

        Ht = H.permute(0, 2, 1)
        covs_measured = torch.bmm(covariance, Ht)

        A = system_covariance.permute(0, 2, 1)
        B = covs_measured.permute(0, 2, 1)
        Kt, _ = torch.gesv(B, A)
        K = Kt.permute(0, 2, 1)

        return K

    @classmethod
    def concatenate_over_time(cls, state_beliefs: Sequence['Gaussian'], design: Design) -> 'GaussianOverTime':
        return GaussianOverTime(state_beliefs=state_beliefs, design=design)

    def sample_transition(self, eps: Optional[Tensor] = None) -> Tensor:
        distribution = torch.distributions.MultivariateNormal(loc=self.means, covariance_matrix=self.covs)
        return deterministic_sample_mvnorm(distribution, eps=eps)

    @property
    def measured_means(self):
        if self._measured_means is None:
            self._measured_means = self.H.matmul(self.means.unsqueeze(2)).squeeze(2)
        return self._measured_means

    @property
    def system_uncertainty(self):
        if self._system_uncertainty is None:
            Ht = self.H.permute(0, 2, 1)
            self._system_uncertainty = self.H.matmul(self.covs).matmul(Ht) + self.R
        return self._system_uncertainty


class GaussianOverTime(StateBeliefOverTime):
    def __init__(self, state_beliefs: Sequence['StateBelief'], design: Design):
        super().__init__(state_beliefs=state_beliefs, design=design)
        means_covs = [(sb.measured_means, sb.system_uncertainty) for sb in self.state_beliefs]
        self.measured_means, self.system_uncertainty = (torch.stack(x, 1) for x in zip(*means_covs))

    def sample_measurements(self, eps: Optional[Tensor] = None) -> Tensor:
        distribution = torch.distributions.MultivariateNormal(self.measured_means, self.system_uncertainty)
        return deterministic_sample_mvnorm(distribution, eps=eps)

    def _log_prob(self, obs: Tensor, subset: Optional[Tuple[Selector, Selector, Selector]] = None) -> Tensor:
        if subset is None:
            subset = (slice(None), slice(None), slice(None))
        m = self.measured_means[subset]
        c = self.system_uncertainty[subset][:, :, :, subset[-1]]
        dist = torch.distributions.MultivariateNormal(m, c)
        return dist.log_prob(obs[subset])
