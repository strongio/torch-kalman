import torch
from torch import Tensor
from torch.nn import Parameter


class Bounded:
    def __init__(self, lower: float, upper: float):
        assert lower >= 0.
        assert upper <= 1.
        self.lower = lower
        self.range = upper - lower
        self.parameter = Parameter(torch.randn(1))

    @property
    def value(self) -> Tensor:
        return torch.sigmoid(self.parameter) * self.range + self.lower
