from typing import Dict, Union, Any, Callable, Iterable, Tuple

from torch import Tensor


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


def split_flat(tens: Tensor, dim: int, clone: bool = False):
    if clone:
        return [tens.select(dim, i).clone() for i in range(tens.shape[dim])]
    else:
        assert not tens.requires_grad
        return [tens.select(dim, i) for i in range(tens.shape[dim])]


def identity(x: Any) -> Any:
    return x


def is_slow_grad(tens: Tensor) -> bool:
    if tens.requires_grad:
        avoid_funs = {'CopyBackwards', 'SelectBackward'}
        next_fun = tens.grad_fn.next_functions[0][0]
        if (tens.grad_fn.__class__.__name__ in avoid_funs) or (next_fun.__class__.__name__ in avoid_funs):
            return True
    return False
