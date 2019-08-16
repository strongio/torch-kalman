import functools
from typing import Callable


def handle_for_batch_kwargs(for_batch_method: Callable) -> Callable:
    """
    Decorates a process's `for_batch` method so that it finds the keyword-arguments that were meant for it.

    :param for_batch_method: The process's `for_batch` method.
    :return: Decorated version.
    """

    @functools.wraps(for_batch_method)
    def wrapped(self, *args, **kwargs):
        new_kwargs = {key: kwargs[key] for key in ('num_groups', 'num_timesteps') if key in kwargs}

        # first, look for bare kwargs:
        for key in self.for_batch_kwargs:
            if key in kwargs:
                new_kwargs[key] = kwargs[key]

        # then, look for process-specific (so will override general):
        for key in self.for_batch_kwargs:
            key2 = "{}.{}".format(self.id, key)
            if key2 in kwargs:
                new_kwargs[key] = kwargs[key2]

        return for_batch_method(self, *args, **new_kwargs)

    wrapped.decorated_pfb = True

    return wrapped
