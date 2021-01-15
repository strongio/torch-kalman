from typing import Tuple, Optional
from warnings import warn

import torch
from torch import Tensor, nn


class SingleOutput(nn.Module):
    def __init__(self, transform: Optional[torch.nn.Module] = None):
        super(SingleOutput, self).__init__()
        self.param = nn.Parameter(.1 * torch.randn(1))
        self.transform = transform

    def forward(self, input: Optional[Tensor] = None) -> Tensor:
        if not (input is None or input.numel() == 0):
            warn(f"{self} is ignoring `input`")
        out = self.param
        if self.transform is not None:
            out = self.transform(out)
        return out


class Identity(nn.Module):
    """
    Identity function
    """

    def forward(self, input: Tensor) -> Tensor:
        return input


class Bounded(nn.Module):
    def __init__(self, value: Tuple[float, float]):
        super(Bounded, self).__init__()
        self.lower, self.upper = value

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input) * (self.upper - self.lower) + self.lower
