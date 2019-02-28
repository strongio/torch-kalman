from collections import defaultdict
from typing import Tuple

import torch

from torch import Tensor
from torch.distributions import Distribution

import numpy as np


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
