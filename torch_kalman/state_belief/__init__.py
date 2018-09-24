from typing import Tuple, Sequence, Union, List

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution

from torch_kalman.state_belief.over_time import GaussianOverTime
from torch_kalman.torch_utils import batch_inverse


# noinspection PyPep8Naming
class StateBelief:
    def __init__(self, means: Tensor, covs: Tensor):
        """
        Belief in the state of the system at a particular timepoint, for a batch of time-series.

        :param means: The means (2D tensor)
        :param covs: The covariances (3D tensor).
        """
        assert means.dim() == 2, "mean should be 2D (first dimension batch-size)"
        assert covs.dim() == 3, "cov should be 3D (first dimension batch-size)"
        if (means != means).any():
            raise ValueError("Missing values in StateBelief (can be caused by gradient-issues -> nan initial-state).")

        batch_size, state_size = means.shape
        assert covs.shape[0] == batch_size, "The batch-size (1st dimension) of cov doesn't match that of mean."
        assert covs.shape[1] == covs.shape[2], "The cov should be symmetric in the last two dimensions."
        assert covs.shape[1] == state_size, "The state-size (2nd/3rd dimension) of cov doesn't match that of mean."

        self.batch_size = batch_size
        self.means = means
        self.covs = covs
        self._H = None
        self._R = None
        self._measurement = None

    def compute_measurement(self, H: Tensor, R: Tensor) -> None:
        if self._measurement is None:
            self._H = H
            self._R = R
        else:
            raise ValueError("`compute_measurement` has already been called for this object")

    @property
    def H(self) -> Tensor:
        if self._H is None:
            raise ValueError("This StateBelief hasn't been measured; use the `measure_state` method.")
        return self._H

    @property
    def R(self) -> Tensor:
        if self._R is None:
            raise ValueError("This StateBelief hasn't been measured; use the `measure_state` method.")
        return self._R

    @property
    def measurement(self) -> Tuple[Tensor, Tensor]:
        if self._measurement is None:
            measured_means = torch.bmm(self.H, self.means[:, :, None]).squeeze(2)
            Ht = self.H.permute(0, 2, 1)
            measured_covs = torch.bmm(torch.bmm(self.H, self.covs), Ht) + self.R
            self._measurement = measured_means, measured_covs
        return self._measurement

    def predict(self, F: Tensor, Q: Tensor) -> 'StateBelief':
        raise NotImplementedError

    def update(self, obs: Tensor) -> 'StateBelief':
        raise NotImplementedError

    @classmethod
    def concatenate_over_time(cls, state_beliefs: Sequence['StateBelief']) -> Distribution:
        raise NotImplementedError()


# noinspection PyPep8Naming
class Gaussian(StateBelief):

    def predict(self, F: Tensor, Q: Tensor) -> StateBelief:
        Ft = F.permute(0, 2, 1)
        means = torch.bmm(F, self.means[:, :, None]).squeeze(2)
        covs = torch.bmm(torch.bmm(F, self.covs), Ft) + Q
        return self.__class__(means=means, covs=covs)

    def update(self, obs: Tensor) -> StateBelief:
        assert isinstance(obs, Tensor)
        measured_means, system_covariance = self.measurement

        # residual:
        residuals = obs - measured_means

        # kalman-gain:
        Sinv = batch_inverse(system_covariance)
        Ht = self.H.permute(0, 2, 1)
        K = torch.bmm(torch.bmm(self.covs, Ht), Sinv)  # kalman gain

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
                       valid_idx: Union[Tensor, List[int], np.ndarray],
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
        I = torch.eye(rank, rank).expand(len(covariance), -1, -1)
        p1 = (I - torch.bmm(K, H))
        p2 = torch.bmm(torch.bmm(p1, covariance), p1.permute(0, 2, 1))
        p3 = torch.bmm(torch.bmm(K, R), K.permute(0, 2, 1))
        return p2 + p3

    @classmethod
    def concatenate_over_time(cls, state_beliefs: Sequence['Gaussian']) -> 'GaussianOverTime':
        return GaussianOverTime(state_beliefs=state_beliefs)