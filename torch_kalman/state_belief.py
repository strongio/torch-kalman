from typing import Tuple, Sequence

import torch

from torch import Tensor

from torch_kalman.torch_utils import batch_inverse

from torch.distributions import MultivariateNormal, Distribution


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

        batch_size, state_size = means.shape
        assert covs.shape[0] == batch_size, "The batch-size (1st dimension) of cov doesn't match that of mean."
        assert covs.shape[1] == covs.shape[2], "The cov should be symmetric in the last two dimensions."
        assert covs.shape[1] == state_size, "The state-size (2nd/3rd dimension) of cov doesn't match that of mean."

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
    def concatenate_over_time(cls, state_beliefs: Sequence) -> Distribution:
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
        residual = obs - measured_means

        # kalman-gain:
        Sinv = batch_inverse(system_covariance)
        Ht = self.H.permute(0, 2, 1)
        K = torch.bmm(torch.bmm(self.covs, Ht), Sinv)  # kalman gain

        # clone tensors since autograd can't handle in-place changes
        means_new = self.means.clone()
        covs_new = self.covs.clone()

        # handle kalman-update for groups w/missing values:
        isnan = (residual != residual)
        groups_with_nan = [i for i in range(len(obs)) if isnan[i].data.any()]  # TODO: can avoid list-comprehension
        if groups_with_nan:
            raise NotImplementedError("TODO: Handle missing valuees")

        # faster kalman-update for groups w/o missing values
        no_nan = [i for i in range(len(obs)) if i not in groups_with_nan]
        if len(no_nan) > 0:
            means_new[no_nan] = self.means[no_nan] + torch.bmm(K[no_nan], residual[no_nan].unsqueeze(2)).squeeze(2)
            covs_new[no_nan] = self.covariance_update(self.covs[no_nan], K[no_nan], self.H[no_nan], self.R[no_nan])

        return self.__class__(means=means_new, covs=covs_new)

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
    def concatenate_over_time(cls, state_beliefs: Sequence[StateBelief]) -> 'GaussianOverTime':
        return GaussianOverTime(state_beliefs=state_beliefs)


class StateBeliefOverTime:
    def __init__(self, state_beliefs: Sequence[StateBelief]):
        """
        Belief in the state of the system over a range of times, for a batch of time-series.

        :param state_beliefs: A sequence of StateBeliefs, ordered chronologically.
        """
        self.state_beliefs = state_beliefs
        self._state_distribution = None
        self._measurement_distribution = None

    def log_prob(self, measurements):
        return self.measurement_distribution.log_prob(measurements)

    @property
    def family(self):
        raise NotImplementedError

    @property
    def measurements(self) -> Tensor:
        # noinspection PyUnresolvedReferences
        return self.measurement_distribution.loc

    @property
    def state_distribution(self) -> Distribution:
        if self._state_distribution is None:
            means, covs = zip(*[(state_belief.means, state_belief.covs) for state_belief in self.state_beliefs])
            means = torch.stack(means).permute(1, 0, 2)
            covs = torch.stack(covs).permute(1, 0, 2, 3)
            self._state_distribution = self.family(loc=means, covariance_matrix=covs)
        return self._state_distribution

    @property
    def measurement_distribution(self) -> Distribution:
        if self._measurement_distribution is None:
            means, covs = zip(*[state_belief.measurement for state_belief in self.state_beliefs])
            means = torch.stack(means).permute(1, 0, 2)
            covs = torch.stack(covs).permute(1, 0, 2, 3)
            self._measurement_distribution = self.family(loc=means, covariance_matrix=covs)
        return self._measurement_distribution


class GaussianOverTime(StateBeliefOverTime):
    @property
    def family(self):
        return MultivariateNormal
