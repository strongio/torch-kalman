import torch
from torch import Tensor
from torch.nn import Parameter


class Bounded:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.range = upper - lower
        self.parameter = Parameter(torch.randn(1))

    def get_value(self) -> Tensor:
        return torch.sigmoid(self.parameter) * self.range + self.lower
