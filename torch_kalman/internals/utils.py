import inspect
from typing import Union, Any, Callable, Iterable, Tuple, Sequence

import torch

import numpy as np


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


def split_flat(tens: torch.Tensor, dim: int, clone: bool = False):
    if clone:
        return [tens.select(dim, i).clone() for i in range(tens.shape[dim])]
    else:
        assert not tens.requires_grad
        return [tens.select(dim, i) for i in range(tens.shape[dim])]


def identity(x: Any) -> Any:
    return x


def is_slow_grad(tens: torch.Tensor) -> bool:
    if tens.requires_grad:
        avoid_funs = {'CopyBackwards', 'SelectBackward'}
        next_fun = tens.grad_fn.next_functions[0][0]
        if (tens.grad_fn.__class__.__name__ in avoid_funs) or (next_fun.__class__.__name__ in avoid_funs):
            return True
    return False


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


def infer_forward_kwargs(nn: Union[torch.nn.Module, Callable], method_name: 'str' = 'forward') -> Sequence[str]:
    if hasattr(nn, '_forward_kwargs'):
        return nn._forward_kwargs
    method = getattr(nn, method_name, None)
    try:
        params = [kwarg for kwarg in inspect.signature(method).parameters if kwarg not in {'self', 'kwargs'}]
    except TypeError as e:
        if e.args[0].endswith('None is not a callable object'):
            params = []
        else:
            raise e
    if not params:
        if method_name == '__call__':
            raise TypeError(
                f"Unable to infer arguments for {nn}. Make sure the `forward` method uses named keyword-arguments."
            )
        return infer_forward_kwargs(nn, method_name='__call__')
    if 'args' in params:
        raise TypeError(
            f"Unable to infer arguments for {nn}. Make sure it does not use `*args, **kwargs`"
        )
    return params
