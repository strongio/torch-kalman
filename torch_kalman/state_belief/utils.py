from collections import defaultdict

import torch

from torch import Tensor
from torch.distributions import Distribution

import numpy as np


def log_prob_with_missings(distribution: Distribution, obs: Tensor) -> Tensor:
    if obs.dim() not in (2, 3):
        raise NotImplementedError("Only implemented for 2/3D tensors")
    elif obs.dim() == 2:
        squeeze_at_end = True
        obs = obs[None, :, :]
    else:
        squeeze_at_end = False

    # remove nans first, see https://github.com/pytorch/pytorch/issues/9688
    isnan = torch.isnan(obs.detach())
    obs = obs.clone()
    obs[isnan] = 0.

    # log-prob:
    out = distribution.log_prob(obs)

    # loop through, looking for partial observations that need to be replaced with a lower-dimensional version:
    partial_idx = defaultdict(list)
    partial_means = defaultdict(list)
    partial_covs = defaultdict(list)
    num_groups, num_times, num_dist_dims = obs.shape
    for t in range(num_times):
        if not isnan[:, t, :].any():
            continue
        for g in range(num_groups):
            this_isnan = isnan[g, t, :]
            if this_isnan.all():
                out[g, t] = 0.
                continue

            valid_idx = tuple(np.where(1 - this_isnan.numpy())[0].tolist())
            partial_idx[valid_idx].append((g, t, valid_idx))
            partial_means[valid_idx].append(distribution.loc[g, t][~this_isnan])
            partial_covs[valid_idx].append(distribution.covariance_matrix[g, t, valid_idx][:, ~this_isnan])

    # do the replacing, minimizing the number of calls to distribution.__init__
    for k, idxs in partial_idx.items():
        dist = distribution.__class__(torch.stack(partial_means[k]), torch.stack(partial_covs[k]))
        partial_obs = torch.stack([obs[idx] for idx in idxs])
        log_probs = dist.log_prob(partial_obs)
        for idx, lp in zip(idxs, log_probs):
            out[idx[:-1]] = lp

    if squeeze_at_end:
        out = out.squeeze(0)

    return out
