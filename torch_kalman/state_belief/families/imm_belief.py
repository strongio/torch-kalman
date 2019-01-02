import itertools
from typing import Optional, Sequence, Tuple, TypeVar

import torch

from numpy.core.multiarray import ndarray
from torch import Tensor
from torch.distributions import Distribution

from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief, GaussianOverTime

from torch_kalman.state_belief.families.gaussian import MultivariateNormal


# noinspection PyPep8Naming
class IMMBelief(StateBelief):
    def __init__(self,
                 state_beliefs: Sequence[StateBelief],
                 mode_probs: Tensor,
                 transition_probs: Tensor):
        self.state_beliefs = state_beliefs
        self.mode_probs = mode_probs  # aka mu
        self.transition_probs = transition_probs  # aka M

        self._marginal_probs = None
        self._mixing_probs = None

        means, covs = self.compute_mixture(weights=self.mode_probs)
        super().__init__(means=means, covs=covs, last_measured=self.state_beliefs[0].last_measured)

    def compute_mixture(self, weights: Tensor) -> Tuple[Tensor, Tensor]:
        means = []
        for w, sb in zip(weights, self.state_beliefs):
            means.append(w * sb.means)
        means = torch.sum(torch.stack(means), 0)

        covs = []
        for w, sb in zip(weights, self.state_beliefs):
            diff = sb.means - means
            outer = (diff.unsqueeze(2) * diff.unsqueeze(1))
            covs.append(w * (outer + sb.covs))
        covs = torch.sum(torch.stack(covs), 0)

        return means, covs

    @property
    def marginal_probs(self) -> Tensor:
        """
        aka cbar
        """
        if self._marginal_probs is None:
            self._marginal_probs = self.mode_probs.matmul(self.transition_probs)
        return self._marginal_probs

    @property
    def mixing_probs(self) -> Tensor:
        """
        aka omega
        """
        if self._mixing_probs is None:
            n = len(self.state_beliefs)
            self._mixing_probs = torch.empty((n, n), device=self.means.device)
            for i, j in itertools.product(range(n), range(n)):
                self._mixing_probs[i, j] = (self.transition_probs[i, j] * self.mode_probs[i]) / self.marginal_probs[j]
        return self._mixing_probs

    def predict(self, F: Tensor, Q: Tensor) -> 'StateBelief':
        predictions = []
        for i, sb in enumerate(self.state_beliefs):
            means, covs = self.compute_mixture(weights=self.mixing_probs[:, i])
            sb_mixed = sb.__class__(means=means, covs=covs, last_measured=sb.last_measured)  # updated inside predict below
            predictions.append(sb_mixed.predict(F=F[i], Q=Q[i]))

        return self.__class__(state_beliefs=predictions,
                              mode_probs=self.mode_probs,
                              transition_probs=self.transition_probs)

    def update(self, obs: Tensor) -> 'StateBelief':
        new_sbs = []
        likelihoods = torch.empty(len(self.state_beliefs))
        for i, sb in enumerate(self.state_beliefs):
            new_sb = sb.update(obs=obs)
            new_sb.compute_measurement(H=sb.H, R=sb.R)
            new_sbs.append(new_sb)
            likelihoods[i] = new_sb.log_prob(obs).exp()

        new_mode_probs = self.marginal_probs * likelihoods
        new_mode_probs /= new_mode_probs.sum()

        return self.__class__(state_beliefs=new_sbs,
                              mode_probs=new_mode_probs,
                              transition_probs=self.transition_probs)

    def compute_measurement(self, H: Tensor, R: Tensor):
        super().compute_measurement(H=H, R=R)
        for sb in self.state_beliefs:
            sb.compute_measurement(H=H, R=R)

    @classmethod
    def concatenate_over_time(cls,
                              state_beliefs: Sequence['IMMBelief'],
                              design: Design,
                              start_datetimes: Optional[ndarray] = None) -> 'GaussianOverTime':
        return GaussianOverTime(state_beliefs=state_beliefs,
                                design=design,
                                start_datetimes=start_datetimes)

    @property
    def distribution(self) -> TypeVar('Distribution'):
        return MultivariateNormal
