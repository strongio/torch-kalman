from math import pi

import torch
from torch import Tensor


def fourier_tensor(time: Tensor, seasonal_period: float, K: int) -> Tensor:
    """
    Given an N-dimensional tensor, create an N+2 dimensional tensor with the 2nd to last dimension corresponding to the Ks
    and the last dimension corresponding to sin/cosine.
    """
    out = torch.empty((*time.shape, K, 2))
    base_index = tuple(slice(0, x) for x in time.shape)
    for idx in range(K):
        k = idx + 1
        for sincos in range(2):
            val = 2. * pi * k * time / seasonal_period
            index = base_index + (idx, sincos)
            out[index] = torch.sin(val) if sincos == 0 else torch.cos(val)
    return out
