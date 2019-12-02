class Batchable:
    _batch_info = None

    @property
    def num_groups(self):
        if self._batch_info is None:
            raise RuntimeError(f"{self} was not created with `for_batch()`, so doesn't have batch-info.")
        return self._batch_info[0]

    @property
    def num_timesteps(self):
        if self._batch_info is None:
            raise RuntimeError(f"{self} was not created with `for_batch()`, so doesn't have batch-info.")
        return self._batch_info[1]

    def for_batch(self, num_groups: int, num_timesteps: int) -> 'Batchable':
        raise NotImplementedError

    @property
    def is_for_batch(self) -> bool:
        return self._batch_info is not None
