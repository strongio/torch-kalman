from collections import defaultdict
from typing import Sequence, Dict, Tuple, Union, Optional
from warnings import warn

import torch

from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief

Selector = Union[Sequence[int], slice]


class StateBeliefOverTime:
    def __init__(self, state_beliefs: Sequence['StateBelief'], design: Design):
        """
        Belief in the state of the system over a range of times, for a batch of time-serieses.
        """
        self.state_beliefs = state_beliefs
        self.design = design
        self.family = self.state_beliefs[0].__class__
        self.num_groups = self.state_beliefs[0].num_groups

        # the last idx where any updates/predicts occurred:
        self._last_update = None
        self._last_prediction = None
        self.last_predict_idx = -torch.ones(self.num_groups, dtype=torch.int)
        self.last_update_idx = -torch.ones(self.num_groups, dtype=torch.int)
        for t, state_belief in enumerate(state_beliefs):
            self.last_predict_idx[state_belief.last_measured == 1] = t
            self.last_update_idx[state_belief.last_measured == 0] = t

        self._means = None
        self._covs = None
        self._last_measured = None

    def _means_covs(self) -> None:
        means, covs = zip(*[(state_belief.means, state_belief.covs) for state_belief in self.state_beliefs])
        self._means = torch.stack(means, 1)
        self._covs = torch.stack(covs, 1)

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

    def components(self) -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:
        states_per_measure = defaultdict(list)
        for state_belief in self.state_beliefs:
            for m, measure in enumerate(self.design.measures):
                H = state_belief.H[:, m, :].data
                m = H * state_belief.means.data
                std = H * torch.diagonal(state_belief.covs.data, dim1=-2, dim2=-1).sqrt()
                states_per_measure[measure].append((m, std))

        out = {}
        for measure, means_and_stds in states_per_measure.items():
            means, stds = zip(*means_and_stds)
            means = torch.stack(means).permute(1, 0, 2)
            stds = torch.stack(stds).permute(1, 0, 2)
            for s, (process_name, state_element) in enumerate(self.design.all_state_elements()):
                if ~torch.isclose(means[:, :, s].abs().max(), torch.zeros(1)):
                    out[(measure, process_name, state_element)] = (means[:, :, s], stds[:, :, s])
        return out

    def last_update(self) -> StateBelief:
        no_updates = self.last_update_idx < 0
        if no_updates.any():
            raise ValueError(f"The following groups have never been updated:\n{no_updates.nonzero().squeeze().tolist()}")
        means_covs = ((self.means[g, t, :], self.covs[g, t, :, :]) for g, t in enumerate(self.last_update_idx))
        means, covs = zip(*means_covs)
        return self.family(means=torch.stack(means), covs=torch.stack(covs))

    def last_prediction(self) -> StateBelief:
        no_predicts = self.last_predict_idx < 0
        if no_predicts.any():
            raise ValueError(f"The following groups have never been predicted:"
                             f"\n{no_predicts.nonzero().squeeze().tolist()}")
        means_covs = ((self.means[g, t, :], self.covs[g, t, :, :]) for g, t in enumerate(self.last_predict_idx))
        means, covs = zip(*means_covs)
        return self.family(means=torch.stack(means), covs=torch.stack(covs))

    def state_belief_for_time(self, times: Sequence[int]) -> StateBelief:
        means_covs = ((self.means[g, t, :], self.covs[g, t, :, :]) for g, t in enumerate(times))
        means, covs = zip(*means_covs)
        return self.family(means=torch.stack(means), covs=torch.stack(covs))

    def log_prob(self, obs: Tensor) -> Tensor:
        if obs.grad_fn is not None:
            warn("`obs` has a grad_fn, nans may propagate to gradient")

        num_groups, num_times, num_dist_dims, *_ = obs.shape

        all_valid_times = list()
        last_nonan_t = -1
        lp_groups = defaultdict(list)
        for t in range(num_times):
            if self._is_nan(obs[:, t]).all():
                # no log-prob needed
                continue

            if not self._is_nan(obs[:, t]).any():
                # will be updated as block:
                if last_nonan_t == (t - 1):
                    last_nonan_t += 1
                else:
                    all_valid_times.append(t)
                continue

            for g in range(num_groups):
                gt_is_nan = self._is_nan(obs[g, t])
                lp_groups[tuple(gt_is_nan.nonzero().tolist())].append((g, t))

        # from [(g1,t1), (g2, t2)] to [(g1,g2), (t1,t2)]
        lp_groups = [(which_valid_idx, tuple(zip(*gt_idx))) for which_valid_idx, gt_idx in lp_groups.items()]

        # shortcuts:
        if last_nonan_t >= 0:
            gt_idx = (slice(None), slice(last_nonan_t + 1))
            lp_groups.append((slice(None), gt_idx))
        for t in all_valid_times:
            gt_idx = (slice(None), t)
            lp_groups.append((slice(None), gt_idx))

        # compute log-probs by dims available:
        out = torch.zeros((num_groups, num_times))
        for which_valid_idx, (group_idx, time_idx) in lp_groups:
            out[group_idx, time_idx] = self._log_prob(obs, subset=(group_idx, time_idx, which_valid_idx))

        return out

    def _log_prob(self, obs: Tensor, subset: Optional[Tuple[Selector, Selector, Selector]] = None):
        raise NotImplementedError

    def _is_nan(self, x: Tensor) -> Tensor:
        return torch.isnan(x)

    def sample_measurements(self, eps: Optional[Tensor] = None):
        raise NotImplementedError
