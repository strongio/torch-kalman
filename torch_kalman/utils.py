from math import pi

import torch
from torch import Tensor


def fourier_series(time: Tensor, seasonal_period: int, parameters: Tensor):
    dim1, dim2 = parameters.shape
    assert dim2 == 2, f"Expected K X 2 matrix, got {(dim1, dim2)}."

    out = torch.zeros_like(time)
    for idx in range(dim1):
        k = idx + 1
        out += (parameters[idx, 0] * torch.sin(2. * pi * k * time / seasonal_period))
        out += (parameters[idx, 1] * torch.cos(2. * pi * k * time / seasonal_period))

    return out
