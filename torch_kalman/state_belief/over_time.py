from collections import defaultdict
from pdb import Pdb
from typing import Sequence, Optional, Dict, Tuple

import torch

from torch import Tensor
from torch.distributions import Distribution

from torch_kalman.design import Design
import numpy as np


class StateBeliefOverTime:
    def __init__(self, state_beliefs: Sequence['StateBelief'], design: Optional[Design] = None):
        """
        Belief in the state of the system over a range of times, for a batch of time-series.

        :param state_beliefs: A sequence of StateBeliefs, ordered chronologically.
        """
        self.state_beliefs = state_beliefs
        self._state_distribution = None
        self._measurement_distribution = None
        self.design = design

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
                    dist = self.distribution(loc=loc, covariance_matrix=cov)
                    out[g, t] = dist.log_prob(measurements[g, t, ~this_isnan])
        return out

    @property
    def distribution(self):
        raise NotImplementedError

    @property
    def measurements(self) -> Tensor:
        # noinspection PyUnresolvedReferences
        return self.measurement_distribution.loc

    def partial_measurements(self, processes: Sequence[str], exclude: bool = True):
        process_idx = self.design.process_idx()
        assert all(process_id in process_idx.keys() for process_id in processes), "Not all `processes` in design."
        if exclude:
            processes = set(process_idx.keys()) - set(processes)

        measurements = []
        covariances = []
        for state_belief in self.state_beliefs:
            # only some states contribute:
            H_partial = torch.zeros_like(state_belief.H)
            for process_id in processes:
                H_partial[:, :, process_idx[process_id]] = state_belief.H[:, :, process_idx[process_id]]

            # mean:
            measurements.append(torch.bmm(H_partial, state_belief.means[:, :, None]).squeeze(2))

            # cov:
            Ht_partial = H_partial.permute(0, 2, 1)
            covariances.append(torch.bmm(torch.bmm(H_partial, state_belief.covs), Ht_partial) + state_belief.R)

        return torch.stack(measurements).permute(1, 0, 2), torch.stack(covariances).permute(1, 0, 2, 3)

    @property
    def state_distribution(self) -> Distribution:
        if self._state_distribution is None:
            means, covs = zip(*[(state_belief.means, state_belief.covs) for state_belief in self.state_beliefs])
            means = torch.stack(means).permute(1, 0, 2)
            covs = torch.stack(covs).permute(1, 0, 2, 3)
            self._state_distribution = self.distribution(loc=means, covariance_matrix=covs)
        return self._state_distribution

    @property
    def measurement_distribution(self) -> Distribution:
        if self._measurement_distribution is None:
            means, covs = zip(*[state_belief.measurement for state_belief in self.state_beliefs])
            means = torch.stack(means).permute(1, 0, 2)
            covs = torch.stack(covs).permute(1, 0, 2, 3)
            self._measurement_distribution = self.distribution(loc=means, covariance_matrix=covs)
        return self._measurement_distribution

    def components(self, design: Optional[Design] = None) -> Dict[Tuple[str, str, str], Tensor]:
        if not design:
            if not self.design:
                raise Exception("Must pass design to `components` if `design` was not passed at init.")
            design = self.design

        states_per_measure = defaultdict(list)
        for state_belief in self.state_beliefs:
            for m, measure in enumerate(design.measures):
                states_per_measure[measure].append(state_belief.H[:, m, :] * state_belief.means.data)

        out = {}
        for measure, tens in states_per_measure.items():
            tens = torch.stack(tens).permute(1, 0, 2)
            for s, (process_name, state_element) in enumerate(design.all_state_elements()):
                if ~torch.isclose(tens[:, :, s].abs().max(), torch.zeros(1)):
                    out[(measure, process_name, state_element)] = tens[:, :, s]
        return out
