from typing import List, Tuple, Optional

import torch
from torch import jit, Tensor, nn


class Exp(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.exp(input)


class SingleOutput(jit.ScriptModule):
    def __init__(self, transform: Optional[torch.nn.Module] = None):
        super(SingleOutput, self).__init__()
        self.param = nn.Parameter(.1 * torch.randn(1))
        self.transform = transform

    @jit.script_method
    def forward(self) -> Tensor:
        out = self.param
        if self.transform is not None:
            out = self.transform(out)
        return out


class Identity(jit.ScriptModule):
    """
    Identity function
    """

    @jit.script_method
    def forward(self, inputs: List[Tensor]) -> Tensor:
        assert len(inputs) == 1
        return inputs[0]


class ReturnValues(jit.ScriptModule):
    """
    Just return a fixed set of values on each call.
    """

    def __init__(self, values: Tensor):
        super(ReturnValues, self).__init__()
        self.values = values

    @jit.script_method
    def forward(self, inputs: List[Tensor] = ()) -> Tensor:
        return self.values


class SimpleTransition(jit.ScriptModule):
    """
    Useful for transitions which are either 1, or a learnable parameter near but <1.
    """

    def __init__(self, value: Tuple[Optional[float], float]):
        super(SimpleTransition, self).__init__()
        lower, upper = value
        self.lower = None if lower is None else torch.full((1,), lower)
        self.upper = torch.full((1,), upper)
        self.parameter = nn.Parameter(torch.randn(1))

    @jit.script_method
    def forward(self, inputs: List[Tensor]) -> Tensor:
        if self.lower is not None:
            return torch.sigmoid(self.parameter) * (self.upper - self.lower) + self.lower
        else:
            return self.upper
