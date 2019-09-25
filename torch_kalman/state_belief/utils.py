from typing import Tuple, Optional

from torch import Tensor
from torch.distributions import MultivariateNormal

import numpy as np
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal


def bmat_idx(*args) -> Tuple:
    """
    Create indices for tensor assignment that act like slices. E.g., batch[:,[1,2,3],[1,2,3]] does not select the upper
    3x3 sub-matrix over batches, but batch[bmat_idx(slice(None),[1,2,3],[1,2,3])] does.

    :param args: Each arg is a sequence of integers. The first N args can be slices, and the last N args can be slices.
    :return: A tuple that can be used for matrix/tensor-selection.
    """

    if len(args) == 0:
        return ()
    elif isinstance(args[-1], slice):
        # trailing slices can't be passed to np._ix, but can be appended to its results
        return bmat_idx(*args[:-1]) + (args[-1],)
    elif isinstance(args[0], slice):
        # leading slices can't be passed to np._ix, but can be prepended to its results
        return (args[0],) + bmat_idx(*args[1:])
    else:
        if any(isinstance(arg, slice) for arg in args[1:]):
            raise ValueError("Only the first/last contiguous args can be slices, not middle args.")
        return np.ix_(*args)


def deterministic_sample_mvnorm(distribution: MultivariateNormal, eps: Optional[Tensor] = None) -> Tensor:
    if eps is None:
        shape = distribution.batch_shape + distribution.event_shape
        eps = _standard_normal(shape, dtype=distribution.loc.dtype, device=distribution.loc.device)
    else:
        if eps.shape[-len(distribution.event_shape):] != distribution.event_shape:
            raise RuntimeError(f"Expected shape ending in {distribution.event_shape}, got {eps.shape}.")
    return distribution.loc + _batch_mv(distribution._unbroadcasted_scale_tril, eps)
