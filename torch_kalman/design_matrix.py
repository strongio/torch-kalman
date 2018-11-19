from typing import Union, Sequence, Dict, Tuple, Optional

import torch
from torch import Tensor


class TensorOverTime:
    def __init__(self, values: Union[Sequence, Tensor], num_timesteps: int, num_groups: int):
        self.num_timesteps = num_timesteps
        self.num_groups = num_groups

        if isinstance(values, Tensor):
            self.check_tens(values, in_list=False)
            values = [values for _ in range(self.num_timesteps)]
        elif isinstance(values, Sequence):
            assert len(values) == self.num_timesteps
            [self.check_tens(tens, in_list=True) for tens in values]

        self.values = values

    def __getitem__(self, item) -> Tensor:
        return self.values[item]

    def check_tens(self, tens: Tensor, in_list: bool) -> None:
        if tens.numel() != 1:
            if list(tens.shape) != [self.num_groups]:
                msg = ("Expected {listof}1D tensor{plural} {each}with length == num_groups.".
                       format(listof='list of ' if in_list else '',
                              plural='s' if in_list else '',
                              each='each ' if in_list else ''))
                raise ValueError(msg)
        if in_list:
            if tens.requires_grad and tens.grad_fn.__class__.__name__ == 'CopyBackwards':
                raise RuntimeError("Please report this error to the package maintainer.")


class DesignMatOverTime:
    @classmethod
    def from_indices_and_vals(cls, indices_and_values: Union[Dict, Sequence[Tuple]], *args, **kwargs) -> 'DesignMatOverTime':
        dynamic = []
        base = torch.zeros(*args, **kwargs)
        if isinstance(indices_and_values, Dict):
            indices_and_values = indices_and_values.items()
        for (r, c), value in indices_and_values:
            if isinstance(value, TensorOverTime):
                dynamic.append(((r, c), value))
            else:
                base[:, r, c] = value
        return cls(base=base, dynamic=dynamic)

    def __init__(self, base: Tensor, dynamic: Optional[Union[Dict, Sequence[Tuple]]] = None):
        self.base = base

        if dynamic is None:
            dynamic = []
        if isinstance(dynamic, dict):
            dynamic = list(dynamic.items())
        self.dynamic = dynamic

        self.versions = {}

    def __getitem__(self, item: int) -> Tensor:
        if item not in self.versions.keys():
            mat = self.base.clone()
            for (r, c), values in self.dynamic:
                mat[:, r, c] = values[item]
            self.versions[item] = mat
        return self.versions[item]
