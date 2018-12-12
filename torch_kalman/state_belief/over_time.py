from collections import defaultdict
from typing import Sequence, Dict, Tuple, TypeVar, Optional

import torch

from torch import Tensor
from torch.distributions import Distribution

from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief

import numpy as np


class StateBeliefOverTime:
    def __init__(self,
                 state_beliefs: Sequence['StateBelief'],
                 design: Design,
                 start_datetimes: Optional[np.ndarray] = None):
        """
        Belief in the state of the system over a range of times, for a batch of time-serieses.
        """
        self.state_beliefs = state_beliefs
        self.design = design
        self.start_datetimes = start_datetimes
        self.family = self.state_beliefs[0].__class__
        self.batch_size = self.state_beliefs[0].batch_size

        # the last idx where any updates/predicts occurred:
        self._last_update = None
        self._last_prediction = None
        self.last_predict_idx = -torch.ones(self.batch_size, dtype=torch.int)
        self.last_update_idx = -torch.ones(self.batch_size, dtype=torch.int)
        for t, state_belief in enumerate(state_beliefs):
            self.last_predict_idx[state_belief.last_measured == 1] = t
            self.last_update_idx[state_belief.last_measured == 0] = t

        self._state_distribution = None
        self._measurement_distribution = None
        self._means = None
        self._covs = None
        self._last_measured = None

    def _means_covs(self) -> None:
        means, covs = zip(*[(state_belief.means, state_belief.covs) for state_belief in self.state_beliefs])
        self._means = torch.stack(means).permute(1, 0, 2)
        self._covs = torch.stack(covs).permute(1, 0, 2, 3)

    @property
    def means(self) -> Tensor:
        if self._means is None:
            self._means_covs()
        return self._means

    @property
    def covs(self) -> Tensor:
        if self._covs is None:
            self._means_covs()
        return self._covs

    def log_prob(self, measurements: Tensor) -> Tensor:
        isnan = torch.isnan(measurements)
        # remove nans first, required due to https://github.com/pytorch/pytorch/issues/9688
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
                    dist = self.distribution(loc=loc, covariance_matrix=cov)
                    out[g, t] = dist.log_prob(measurements[g, t, ~this_isnan])
        return out

    @property
    def distribution(self) -> TypeVar('Distribution'):
        raise NotImplementedError

    @property
    def measurements(self) -> Tensor:
        # noinspection PyUnresolvedReferences
        return self.measurement_distribution.loc

    def partial_measurements(self, exclude: Sequence[Tuple[str, str]]) -> Tuple[Tensor, Tensor]:
        remaining = set(exclude)
        exclude_idx = []
        for i, ps in enumerate(self.design.all_state_elements()):
            if ps in exclude:
                exclude_idx.append(i)
                remaining.remove(ps)
        if remaining:
            raise ValueError(f"The following were in `exclude` but weren't found in design:\n{remaining}.")

        measurements = []
        covariances = []
        for state_belief in self.state_beliefs:
            # only some states contribute:
            H_partial = state_belief.H.clone()
            H_partial[:, :, exclude_idx] = 0.

            # mean:
            measurements.append(torch.bmm(H_partial, state_belief.means[:, :, None]).squeeze(2))

            # cov:
            Ht_partial = H_partial.permute(0, 2, 1)
            covariances.append(torch.bmm(torch.bmm(H_partial, state_belief.covs), Ht_partial) + state_belief.R)

        return torch.stack(measurements).permute(1, 0, 2), torch.stack(covariances).permute(1, 0, 2, 3)

    @property
    def state_distribution(self) -> Distribution:
        if self._state_distribution is None:
            self._state_distribution = self.distribution(loc=self.means, covariance_matrix=self.covs)
        return self._state_distribution

    @property
    def measurement_distribution(self) -> Distribution:
        if self._measurement_distribution is None:
            means, covs = zip(*[state_belief.measurement for state_belief in self.state_beliefs])
            means = torch.stack(means).permute(1, 0, 2)
            covs = torch.stack(covs).permute(1, 0, 2, 3)
            self._measurement_distribution = self.distribution(loc=means, covariance_matrix=covs)
        return self._measurement_distribution

    def components(self) -> Dict[Tuple[str, str, str], Tensor]:
        states_per_measure = defaultdict(list)
        for state_belief in self.state_beliefs:
            for m, measure in enumerate(self.design.measures):
                states_per_measure[measure].append(state_belief.H[:, m, :] * state_belief.means.data)

        out = {}
        for measure, tens in states_per_measure.items():
            tens = torch.stack(tens).permute(1, 0, 2)
            for s, (process_name, state_element) in enumerate(self.design.all_state_elements()):
                if ~torch.isclose(tens[:, :, s].abs().max(), torch.zeros(1)):
                    out[(measure, process_name, state_element)] = tens[:, :, s]
        return out

    @property
    def last_update(self) -> StateBelief:
        if self._last_update is None:
            no_updates = self.last_update_idx < 0
            if no_updates.any():
                raise ValueError(f"The following groups have never been updated:\n{no_updates.nonzero().squeeze().tolist()}")
            means_covs = ((self.means[g, t, :], self.covs[g, t, :, :]) for g, t in enumerate(self.last_update_idx))
            means, covs = zip(*means_covs)
            self._last_update = self.family(means=torch.stack(means), covs=torch.stack(covs))
        return self._last_update

    @property
    def last_prediction(self) -> StateBelief:
        if self._last_prediction is None:
            no_predicts = self.last_predict_idx < 0
            if no_predicts.any():
                raise ValueError(f"The following groups have never been predicted:"
                                 f"\n{no_predicts.nonzero().squeeze().tolist()}")
            means_covs = ((self.means[g, t, :], self.covs[g, t, :, :]) for g, t in enumerate(self.last_predict_idx))
            means, covs = zip(*means_covs)
            self._last_prediction = self.family(means=torch.stack(means), covs=torch.stack(covs))
        return self._last_prediction

    def slice_by_dt(self, datetimes: np.ndarray) -> StateBelief:
        if self.start_datetimes is None:
            raise ValueError("Cannot use `slice_by_dt` if `start_datetimes` was not passed originally.")
        act = datetimes.dtype
        exp = self.start_datetimes.dtype
        assert act == exp, f"Expected datetimes with dtype {exp}, got {act}."
        idx = (datetimes - self.start_datetimes).view('int64')
        means_covs = ((self.means[g, t, :], self.covs[g, t, :, :]) for g, t in enumerate(idx))
        means, covs = zip(*means_covs)
        return self.family(means=torch.stack(means), covs=torch.stack(covs))
