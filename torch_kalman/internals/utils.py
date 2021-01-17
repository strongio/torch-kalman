from typing import Union, Any, Callable, Iterable, Tuple, Sequence

import torch

import numpy as np


def get_owned_kwarg(owner: str, key: str, kwargs: dict) -> tuple:
    specific_key = f"{owner}__{key}"
    if specific_key in kwargs:
        return specific_key, kwargs[specific_key]
    elif key in kwargs:
        return key, kwargs[key]
    else:
        raise TypeError(f"Missing required keyword-arg `{key}` (or `{specific_key}`).")


def bifurcate(x: Iterable, lhs: Callable[[Any], bool]) -> Tuple[list, list]:
    """
    Split an iterable into two lists depending on a condition.

    :param x: An iterable.
    :param lhs: A function that takes an element of x; when this returns True, the element is added to the left output,
      when this returns False, the element is added to the right output.
    :return: Two lists.
    """
    l, r = [], []
    for el in x:
        if lhs(el):
            l.append(el)
        else:
            r.append(el)
    return l, r


def zpad(x: Any, n: int) -> str:
    return str(x).rjust(n, "0")


def identity(x: Any) -> Any:
    return x


def ragged_cat(tensors: Sequence[torch.Tensor], ragged_dim: int, cat_dim: int = 0) -> torch.Tensor:
    max_dim_len = max(tensor.shape[ragged_dim] for tensor in tensors)
    out = []
    num_dims = len(tensors[0].shape)
    for tensor in tensors:
        this_tens_dim_len = tensor.shape[ragged_dim]
        shape = list(tensor.shape)
        assert len(shape) == num_dims
        shape[ragged_dim] = max_dim_len
        padded = torch.empty(shape)
        padded[:] = float('nan')
        idx = tuple(slice(0, this_tens_dim_len) if i == ragged_dim else slice(None) for i in range(num_dims))
        padded[idx] = tensor
        out.append(padded)
    return torch.cat(out, cat_dim)


def true1d_idx(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()
    arr = arr.astype('bool')
    if len(arr.shape) > 1:
        raise ValueError("Expected 1d array.")
    return np.where(arr)[0]
