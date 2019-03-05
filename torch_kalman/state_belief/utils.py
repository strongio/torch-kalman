from typing import Tuple, Optional

import torch

from torch import Tensor
from torch.distributions import MultivariateNormal

import numpy as np
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal


def bmat_idx(*args) -> Tuple:
    """
    Create indices for tensor assignment that act like slices. E.g., batch[:,[1,2,3],[1,2,3]] does not select the upper
    3x3 sub-matrix over batches, but batch[_bmat_idx(slice(None),[1,2,3],[1,2,3])] does.

    :param args: Each arg is a sequence of integers. The first N args can be slices, and the last N args can be slices.
    :return: A tuple that can be used for matrix/tensor-selection.
    """
    # trailing slices can simply be removed:
    up_to_idx = len(args)
    for arg in reversed(args):
        if isinstance(arg, slice):
            up_to_idx -= 1
        else:
            break
    args = args[:up_to_idx]

    if len(args) == 0:
        return ()
    elif isinstance(args[0], slice):
        # leading slices can't be passed to np._ix, but can be prepended to it
        return (args[0],) + bmat_idx(*args[1:])
    else:
        return np.ix_(*args)


def deterministic_sample_mvnorm(distribution: MultivariateNormal, eps: Optional[Tensor] = None) -> Tensor:
    expected_shape = distribution._batch_shape + distribution._event_shape
    univariate = len(distribution.event_shape) == 1 and distribution.event_shape[0] == 1
    if univariate:
        if eps is None:
            eps = distribution.loc.new(*expected_shape).normal_()
        else:
            assert eps.size() == expected_shape, f"expected-shape:{expected_shape}, actual:{eps.size()}"
        std = torch.sqrt(torch.squeeze(distribution.covariance_matrix, -1))
        return std * eps + distribution.loc
    else:
        if eps is None:
            eps = _standard_normal(expected_shape, dtype=distribution.loc.dtype, device=distribution.loc.device)
        else:
            assert eps.size() == expected_shape, f"expected-shape:{expected_shape}, actual:{eps.size()}"
        return distribution.loc + _batch_mv(distribution._unbroadcasted_scale_tril, eps)
