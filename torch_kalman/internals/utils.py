from typing import Union, Any, Callable, Iterable, Tuple, Sequence, List, Optional

import torch

import numpy as np


def get_nan_groups(isnan: torch.Tensor) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Iterable of (group_idx, valid_idx) tuples that can be passed to torch.meshgrid. If no valid, then not returned; if
    all valid then (group_idx, None) is returned; can skip call to meshgrid.
    """
    assert len(isnan.shape) == 2
    state_dim = isnan.shape[-1]
    out: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    for nan_combo in torch.unique(isnan, dim=0):
        num_nan = nan_combo.sum()
        if num_nan < state_dim:
            c1 = (isnan * nan_combo[None, :]).sum(1) == num_nan
            c2 = (~isnan * ~nan_combo[None, :]).sum(1) == (state_dim - num_nan)
            group_idx = (c1 & c2).nonzero().view(-1)
            if num_nan == 0:
                valid_idx = None
            else:
                valid_idx = (~nan_combo).nonzero().view(-1)
            out.append((group_idx, valid_idx))
    return out


def get_owned_kwarg(owner: str, key: str, kwargs: dict) -> tuple:
    specific_key = f"{owner}__{key}"
    if specific_key in kwargs:
        return specific_key, kwargs[specific_key]
    elif key in kwargs:
        return key, kwargs[key]
    else:
        raise TypeError(f"Missing required keyword-arg `{key}` (or `{specific_key}`).")


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
