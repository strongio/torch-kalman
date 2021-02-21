from typing import Tuple, Optional, Sequence
from warnings import warn

import torch
from torch import Tensor, nn

from torch_kalman.utils.features import fourier_tensor


class SingleOutput(nn.Module):
    """
    Basically a callable parameter, with optional transform.
    """

    def __init__(self, numel: int = 1, transform: Optional[torch.nn.Module] = None):
        super(SingleOutput, self).__init__()
        self.param = nn.Parameter(.1 * torch.randn(numel))
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
    """
    Transforms input to fall within `value`, a tuple of (lower, upper)
    """

    def __init__(self, lower: float, upper: float):
        super(Bounded, self).__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input) * (self.upper - self.lower) + self.lower


class Multi(nn.Module):
    """
    Multiplies input by `value`
    """

    def __init__(self, value: torch.Tensor):
        super(Multi, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input * self.value


class TimesToFourier(nn.Module):
    """
    Takes a (N x 1) tensor of times and outputs a (N X K*2) tensor of fourier terms, useful for modeling seasonality.
    """

    def __init__(self, K: int, seasonal_period: float):
        super(TimesToFourier, self).__init__()
        self.K = K
        self.seasonal_period = float(seasonal_period)

    def forward(self, times: torch.Tensor):
        if len(times.shape) >= 2:
            assert len(times.shape) == 2 and times.shape[-1] == 1
            times = times.squeeze(-1)
        return fourier_tensor(times, seasonal_period=self.seasonal_period, K=self.K).view(times.shape[0], self.K * 2)


class Assignments(nn.Module):
    """
    Takes a (N x K) input and maps it to a (N X G) output by assigning columns of the input to columns of the output.
    """

    def __init__(self,
                 input_cols: int,
                 output_cols: int,
                 assignments: Sequence[Tuple[int, int]],
                 padding: float = 0.):
        """
        :param input_cols: Number of columns to expect in input.
        :param output_cols: Number of columns to generate in output.
        :param assignments: A list of tuples with (from-col, to-col).
        :param padding: Padding for unassigned output elements, default zero.
        """
        super(Assignments, self).__init__()
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.assignments = list(assignments)
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        assert input.shape[-1] == self.input_cols
        output = torch.full(input.shape[:-1] + (self.output_cols,), fill_value=self.padding)
        for from_, to_ in self.assignments:
            output[:, to_] = input[:, from_]
        return output
