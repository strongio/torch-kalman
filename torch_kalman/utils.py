from math import pi
from typing import Dict, Union, Any

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


def itervalues_sorted_keys(adict: Dict):
    for k in sorted(adict.keys()):
        yield adict[k]


def dict_key_replace(obj: Union[Dict, Any], old: str, new: str) -> Dict:
    if not isinstance(obj, Dict):
        return obj

    out = {}
    for key, value in obj.items():
        if key == old:
            out[new] = dict_key_replace(value, old=old, new=new)
        else:
            out[key] = dict_key_replace(value, old=old, new=new)
    return out


def zpad(x, n):
    return str(x).rjust(n, "0")
