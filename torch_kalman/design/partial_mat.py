from typing import Sequence, Dict, Iterable

import torch

from torch_kalman.covariance import Covariance


class PartialCovariance:
    def __init__(self,
                 full_dim_names: Iterable,
                 partial_dim_names: Iterable,
                 cov_kwargs: Dict):
        names_in_partial_but_not_full = set(partial_dim_names) - set(full_dim_names)
        if len(names_in_partial_but_not_full):
            raise ValueError(f"The following are present in `partial_dim_names` but not `full_dim_names`:"
                             f"\n{names_in_partial_but_not_full}")

        self.full_dim_names = list(full_dim_names)
        self.partial_dim_names = list(partial_dim_names)
        self.cov_kwargs = cov_kwargs

    def _create_partial_cov(self) -> torch.Tensor:
        return Covariance.from_log_cholesky(**self.cov_kwargs)

    def create(self, leading_dims: Sequence[int]) -> torch.Tensor:
        rank = len(self.full_dim_names)
        partial_rank = len(self.partial_dim_names)

        partial_init_cov = self._create_partial_cov()
        if partial_init_cov.shape[-1] != partial_rank:
            raise RuntimeError(f"The cov_kwargs passed to {self.__class__.__name__} do not create a matrix whos last "
                               f"dim == len(partial_dim_names).")

        cov = torch.zeros(size=tuple(leading_dims) + (rank, rank), device=self.cov_kwargs.get('device', None))
        for r in range(partial_rank):
            for c in range(partial_rank):
                to_r = self.full_dim_names.index(self.partial_dim_names[r])
                to_c = self.full_dim_names.index(self.partial_dim_names[c])
                cov[..., to_r, to_c] = partial_init_cov[r, c]

        return cov
