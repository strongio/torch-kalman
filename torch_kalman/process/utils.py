from typing import List, Tuple, Optional

import torch
from torch import Tensor, nn


class SingleOutput(nn.Module):
    def __init__(self, transform: Optional[torch.nn.Module] = None):
        super(SingleOutput, self).__init__()
        self.param = nn.Parameter(.1 * torch.randn(1))
        self.transform = transform

    def forward(self) -> Tensor:
        out = self.param
        if self.transform is not None:
            out = self.transform(out)
        return out


class Identity(nn.Module):
    """
    Identity function
    """

    def forward(self, inputs: List[Tensor]) -> Tensor:
        assert len(inputs) == 1
        return inputs[0]


class ReturnValues(nn.Module):
    """
    Just return a fixed set of values on each call.
    """

    def __init__(self, values: Tensor):
        super(ReturnValues, self).__init__()
        # if len(values.shape) == 1:
        #     values = values.unsqueeze(-1)
        self.values = values

    def forward(self, inputs: List[Tensor] = ()) -> Tensor:
        return self.values


class SimpleTransition(nn.Module):
    """
    Useful for transitions which are either 1, or a learnable parameter near but <1.
    """

    def __init__(self, value: Tuple[Optional[float], float]):
        super(SimpleTransition, self).__init__()
        lower, upper = value
        self.lower = None if lower is None else torch.full((1,), lower)
        self.upper = torch.full((1,), upper)
        self.parameter = nn.Parameter(torch.randn(1))

    def forward(self, inputs: List[Tensor]) -> Tensor:
        if self.lower is not None:
            out = torch.sigmoid(self.parameter) * (self.upper - self.lower) + self.lower
        else:
            out = self.upper
        return out
