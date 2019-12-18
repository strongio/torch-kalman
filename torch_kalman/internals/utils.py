from typing import Dict, Union, Any, Callable, Iterable, Tuple, Sequence

import torch


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
