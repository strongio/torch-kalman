import functools
from typing import Callable


class method_for_batch:
    def __init__(self, setting: bool):
        """
        :param setting: If True, then method can only be called on output of `for_batch`; if False, then method
        *cannot* be called on output of `for_batch`.
        """
        self.setting = setting

    def __call__(self, func: Callable):
        msg = "Can only" if self.setting else "Cannot"

        @functools.wraps(func)
        def wrapped(slf, *args, **kwargs):
            if bool(slf.is_for_batch) != self.setting:
                raise ValueError(
                    f"{msg} call `{type(slf).__name__}.{func.__name__}()` if it's the output of `for_batch()`."
                )
            return func(slf, *args, **kwargs)

        return wrapped
