from typing import Sequence

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from torch_kalman.state_belief import StateBelief


class StateBeliefOverTime:
    def __init__(self, state_beliefs: Sequence[StateBelief]):
        """
        Belief in the state of the system over a range of times, for a batch of time-series.

        :param state_beliefs: A sequence of StateBeliefs, ordered chronologically.
        """
        self.state_beliefs = state_beliefs
        self._state_distribution = None
        self._measurement_distribution = None

    def log_prob(self, measurements: Tensor) -> Tensor:
        isnan = torch.isnan(measurements)
        # remove nans first, required due to bug: https://github.com/pytorch/pytorch/issues/9688
        measurements = measurements.clone()
        measurements[isnan] = 0.

        # check those group x times where only some dimensions were nan; use lower dimensional
        # family to compute log-prob
        num_groups, num_times, num_dims = measurements.shape
        out = self.measurement_distribution.log_prob(measurements)
        for g in range(num_groups):
            for t in range(num_times):
                this_isnan = isnan[g, t, :]
                if this_isnan.any():
                    if this_isnan.all():
                        out[g, t] = 0.
                        continue
                    loc = self.measurement_distribution.loc[g, t][~this_isnan].clone()
                    cov = self.measurement_distribution.covariance_matrix[g, t][~this_isnan][:, ~this_isnan].clone()
                    dist = self.family(loc=loc, covariance_matrix=cov)
                    out[g, t] = dist.log_prob(measurements[g, t, ~this_isnan])
        return out

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