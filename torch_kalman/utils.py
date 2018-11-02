from math import pi

import torch
from numpy import prod
from torch import Tensor


def fourier_series(time: Tensor, seasonal_period: float, K: int) -> Tensor:
    batch_size, *other_dims = time.shape
    if prod(other_dims) > 1.0:
        raise ValueError("`time` should be one-dimensional")
    time = time.squeeze()

    out = torch.empty((batch_size, K, 2))
    for idx in range(K):
        k = idx + 1
        for sincos in range(2):
            val = 2. * pi * k * time / seasonal_period
            out[:, idx, sincos] = torch.sin(val) if sincos == 0 else torch.cos(val)

    return out
