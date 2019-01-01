import itertools
from typing import Optional, Sequence, Tuple

import torch
from numpy.core.multiarray import ndarray
from torch import Tensor
from torch.distributions import Distribution

from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief


# noinspection PyPep8Naming
class IMM(StateBelief):
    def __init__(self,
                 state_beliefs: Sequence[StateBelief],
                 mode_probs: Tensor,
                 transition_probs: Tensor,
                 last_measured: Optional[Tensor] = None):
        self.state_beliefs = state_beliefs
        self.mode_probs = mode_probs  # aka mu
        self.transition_probs = transition_probs  # aka M

        self._marginal_probs = None
        self._mixing_probs = None

        means, covs = self.compute_mixture()
        super().__init__(means=means, covs=covs, last_measured=last_measured)

    def compute_mixture(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @property
    def marginal_probs(self) -> Tensor:
        """
        aka cbar
        """
        if self._marginal_probs is None:
            self._marginal_probs = torch.dot(self.mode_probs, self.transition_probs)
        return self._marginal_probs

    @property
    def mixing_probs(self) -> Tensor:
        """
        aka omega
        """
        if self._mixing_probs is None:
            n = len(self.state_beliefs)
            for i, j in itertools.product(range(n), range(n)):
                self._mixing_probs[i, j] = (self.transition_probs[i, j] * self.mode_probs[i]) / self.marginal_probs[j]
        return self._mixing_probs

    def predict(self, F: Tensor, Q: Tensor) -> 'StateBelief':
        # TODO: separate F and Q for each
        n = len(self.state_beliefs)
        predictions = []
        for i in range(n):
            weights = self.mixing_probs[:, i]

            means = []
            covs = []
            for sb, w in zip(weights, self.state_beliefs):
                means.append(sb.means * w)
                diff = sb.means - self.means
                covs.append(sb.covs * (torch.outer(diff, diff) + sb.covs))
            means = torch.sum(torch.stack(means))
            covs = torch.sum(torch.stack(covs))

            sb_mixed = self.family(means=means, covs=covs, last_measured='TODO')
            predictions.append(sb_mixed.predict(F=F[i], Q=Q[i]))

        return self.__class__(state_beliefs=predictions,
                              mode_probs=self.mode_probs,
                              transition_probs=self.transition_probs,
                              last_measured='TODO')

    def update(self, obs: Tensor) -> 'StateBelief':
        new_sbs = []
        likelihoods = torch.empty(len(self.state_beliefs))
        for i, sb in enumerate(self.state_beliefs):
            new_sb = sb.update(obs=obs)
            new_sbs.append(new_sb)
            likelihoods[i] = new_sb.likelihood  # TODO

        new_mode_probs = self.marginal_probs * likelihoods
        new_mode_probs /= new_mode_probs.sum()

        return self.__class__(state_beliefs=new_sbs,
                              mode_probs=new_mode_probs,
                              transition_probs=self.transition_probs,
                              last_measured='TODO')

    @classmethod
    def concatenate_over_time(cls,
                              state_beliefs: Sequence['StateBelief'],
                              design: Design,
                              start_datetimes: Optional[ndarray] = None) -> 'StateBeliefOverTime':
        raise NotImplementedError

    def to_distribution(self) -> Distribution:
        raise NotImplementedError
