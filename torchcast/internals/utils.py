from typing import Union, Any, Tuple, Sequence, List, Optional

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
    if state_dim == 1:
        # shortcut for univariate
        group_idx = (~isnan.squeeze(-1)).nonzero().view(-1)
        out.append((group_idx, None))
        return out
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


def ragged_cat(tensors: Sequence[torch.Tensor],
               ragged_dim: int,
               cat_dim: int = 0,
               padding: Optional[float] = None) -> torch.Tensor:

    max_dim_len = max(tensor.shape[ragged_dim] for tensor in tensors)
    if padding is None:
        padding = float('nan')
    out = []
    num_dims = len(tensors[0].shape)
    for tensor in tensors:
        this_tens_dim_len = tensor.shape[ragged_dim]
        shape = list(tensor.shape)
        assert len(shape) == num_dims
        shape[ragged_dim] = max_dim_len
        padded = torch.empty(shape, dtype=tensors[0].dtype, device=tensors[0].device)
        padded[:] = padding
        idx = tuple(slice(0, this_tens_dim_len) if i == ragged_dim else slice(None) for i in range(num_dims))
        padded[idx] = tensor
        out.append(padded)
    return torch.cat(out, cat_dim)


@torch.no_grad()
def true1d_idx(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if not isinstance(arr, torch.Tensor):
        arr = torch.as_tensor(arr)
    arr = arr.bool()
    if len(arr.shape) > 1:
        raise ValueError("Expected 1d array.")
    return arr.nonzero(as_tuple=True)[0]


def is_near_zero(tens: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> torch.Tensor:
    z = torch.zeros(1, dtype=tens.dtype, device=tens.device)
    return torch.isclose(tens, other=z, rtol=rtol, atol=atol, equal_nan=equal_nan)
