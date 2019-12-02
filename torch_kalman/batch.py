from typing import Tuple


class Batchable:
    _batch_info = None

    @property
    def batch_info(self):
        return self._batch_info

    @batch_info.setter
    def batch_info(self, value: Tuple[int, int]):
        if self.batch_info:
            raise RuntimeError(f"Cannot set `{self}.batch_info`, was already set to {self.batch_info}.")
        if value[0] < 1 or value[1] < 1 or len(value) > 2:
            raise ValueError(f"Expected a tuple of two positive integers, got {value}.")
        self._batch_info = value

    @property
    def num_groups(self):
        if self.batch_info is None:
            raise RuntimeError(f"{self} was not created with `for_batch()`, so doesn't have batch-info.")
        return self.batch_info[0]

    @property
    def num_timesteps(self):
        if self.batch_info is None:
            raise RuntimeError(f"{self} was not created with `for_batch()`, so doesn't have batch-info.")
        return self.batch_info[1]

    def for_batch(self, num_groups: int, num_timesteps: int) -> 'Batchable':
        raise NotImplementedError

    @property
    def is_for_batch(self) -> bool:
        return self.batch_info is not None
