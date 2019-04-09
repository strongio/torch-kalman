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
        self._H = None
        self._R = None
        self._predictions = None
        self._prediction_uncertainty = None

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

    def _which_valid_key(self, is_nan: Tensor) -> Tuple[int]:
        num_multi_dims = sum(x > 1 for x in is_nan.shape)
        if num_multi_dims > 1:
            raise ValueError("Expected `tensor` to be 1D (or have only one non-singleton dimension.")
        is_valid = ~is_nan
        return tuple(is_valid.nonzero().squeeze(-1).tolist())

    def log_prob(self, obs: Tensor, **kwargs) -> Tensor:
        if obs.grad_fn is not None:
            warn("`obs` has a grad_fn, nans may propagate to gradient")

        num_groups, num_times, num_dist_dims = obs.shape

        # group into chunks for log-prob evaluation. the way indexing works makes this tricky, and slow if we just create a
        # separate group X measure index for each separate time-slice. two shortcuts are used to mitigate this:
        # (1) the first N time-slices that are nan-free will all be evaluated as a chunk
        # (2) subsequent nan-free slices use `slice` notation instead of having to iterate through each group, checking
        #     which measures were nan
        # For all other time-points, we need a separate (group-indices, time-index, measure-indices) tuple.

        times_without_nan = list()
        last_nonan_t = -1
        lp_groups = defaultdict(list)
        for t in range(num_times):
            if torch.isnan(obs[:, t]).all():
                # no log-prob needed
                continue

            if not torch.isnan(obs[:, t]).any():
                # will be updated as block:
                if last_nonan_t == (t - 1):
                    last_nonan_t += 1
                else:
                    times_without_nan.append(t)
                continue

            for g in range(num_groups):
                is_nan = torch.isnan(obs[g, t])
                if is_nan.all():
                    # no log-prob needed
                    continue
                measure_idx = self._which_valid_key(is_nan)
                lp_groups[(t, measure_idx)].append(g)

        lp_groups = [(gidx, t, midx) for (t, midx), gidx in lp_groups.items()]

        # shortcuts:
        if last_nonan_t >= 0:
            gtm = slice(None), slice(last_nonan_t + 1), slice(None)
            lp_groups.append(gtm)
        if len(times_without_nan):
            gtm = slice(None), times_without_nan, slice(None)
            lp_groups.append(gtm)

        # compute log-probs by dims available:
        out = torch.zeros((num_groups, num_times))
        for group_idx, time_idx, measure_idx in lp_groups:
            if isinstance(time_idx, int):
                # assignment is dimensionless in time; needed b/c group isn't a slice
                lp = self._log_prob_with_subsetting(obs, group_idx=group_idx, time_idx=(time_idx,), measure_idx=measure_idx,
                                                    **kwargs)
                out[group_idx, time_idx] = lp.squeeze(-1)
            else:
                # time has dimension, but group is a slice so it's OK
                out[group_idx, time_idx] = \
                    self._log_prob_with_subsetting(obs, group_idx=group_idx, time_idx=time_idx, measure_idx=measure_idx,
                                                   **kwargs)

        return out

    def _log_prob_with_subsetting(self,
                                  obs: Tensor,
                                  group_idx: Selector,
                                  time_idx: Selector,
                                  measure_idx: Selector,
                                  **kwargs) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _check_lp_sub_input(group_idx: Selector, time_idx: Selector):
        if isinstance(group_idx, Sequence) and isinstance(time_idx, Sequence):
            if len(group_idx) > 1 and len(time_idx) > 1:
                warn("Both `group_idx` and `time_idx` are indices (i.e. neither is an int or a slice). This is rarely the "
                     "expected input.")

    def sample_measurements(self, eps: Optional[Tensor] = None):
        raise NotImplementedError

    @property
    def H(self) -> Tensor:
        if self._H is None:
            self._H = torch.stack([sb.H for sb in self.state_beliefs], 1)
        return self._H

    @property
    def R(self) -> Tensor:
        if self._R is None:
            self._R = torch.stack([sb.R for sb in self.state_beliefs], 1)
        return self._R

    @property
    def predictions(self) -> Tensor:
        if self._predictions is None:
            self._predictions = self.H.matmul(self.means.unsqueeze(3)).squeeze(3)
        return self._predictions

    @property
    def prediction_uncertainty(self) -> Tensor:
        if self._prediction_uncertainty is None:
            Ht = self.H.permute(0, 1, 3, 2)
            self._prediction_uncertainty = self.H.matmul(self.covs).matmul(Ht) + self.R
        return self._prediction_uncertainty
