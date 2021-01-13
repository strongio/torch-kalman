from typing import List, Dict, Iterable, Optional

import torch

from torch import Tensor, nn, jit
from torch_kalman.internals.utils import empty_list_of_str


class Covariance(nn.Module):
    def __init__(self,
                 rank: int,
                 empty_idx: List[int] = (),
                 method: str = 'log_cholesky'):
        super(Covariance, self).__init__()
        self.rank = rank
        if len(empty_idx) == 0:
            empty_idx = [self.rank + 1]  # jit doesn't seem to like empty lists
        self.empty_idx = empty_idx
        self.method = method
        if self.method == 'log_cholesky':
            param_rank = len([i for i in range(self.rank) if i not in self.empty_idx])
            self.cholesky_log_diag = torch.nn.Parameter(.1 * torch.randn(param_rank))
            n_off = int(param_rank * (param_rank - 1) / 2)
            self.cholesky_off_diag = torch.nn.Parameter(.1 * torch.randn(n_off))
        else:
            raise NotImplementedError(method)

        #
        self.cache: Dict[str, Tensor] = {'_null': torch.empty(0)}  # jit dislikes empty dicts
        self._cache_enabled = False

        self.expected_kwargs: Optional[List[str]] = None
        self.time_varying_kwargs: Optional[List[str]] = None

    @jit.unused
    def get_all_expected_kwargs(self) -> Iterable[str]:
        return (x for x in self.expected_kwargs or [])

    @jit.unused
    def enable_cache(self, enable: bool = True):
        if enable:
            self.cache.clear()
        self._cache_enabled = enable

    @staticmethod
    def log_chol_to_chol(log_diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
        assert log_diag.shape[:-1] == off_diag.shape[:-1]

        rank = log_diag.shape[-1]
        L = torch.diag_embed(torch.exp(log_diag))

        idx = 0
        for i in range(rank):
            for j in range(i):
                L[..., i, j] = off_diag[..., idx]
                idx += 1
        return L

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        key = self._get_cache_key(inputs)
        if self._cache_enabled and key is not None:
            if key not in self.cache:
                self.cache[key] = self._get_padded_cov(inputs)
            cov = self.cache[key]
        else:
            cov = self._get_padded_cov(inputs)
        return cov

    def _get_cache_key(self, inputs: Dict[str, Tensor]) -> Optional[str]:
        """
        Subclasses could use `inputs` to determine the cache-key
        """
        if len(inputs) == 0:
            return 'static'
        elif self.time_varying_kwargs is not None:
            if any([k in self.time_varying_kwargs for k in inputs]):
                return None
        raise NotImplementedError("TODO")

    def _get_padded_cov(self, inputs: Dict[str, Tensor]) -> Tensor:
        if self.method == 'log_cholesky':
            L = self.log_chol_to_chol(self.cholesky_log_diag, self.cholesky_off_diag)
            # TODO: predicting diagonal multis. ideally cache the base matrix and only recompute multis?
            return pad_covariance(L @ L.t(), [int(i not in self.empty_idx) for i in range(self.rank)])
        else:
            raise NotImplementedError


def pad_covariance(unpadded_cov: Tensor, mask_1d: List[int]) -> Tensor:
    rank = len(mask_1d)
    padded_to_unpadded: Dict[int, int] = {}
    up_idx = 0
    for p_idx, is_filled in enumerate(mask_1d):
        if is_filled == 1:
            padded_to_unpadded[p_idx] = up_idx
            up_idx += 1
    if up_idx == len(mask_1d):
        # shortcut
        return unpadded_cov

    out = torch.zeros(rank, rank)
    for to_r in range(rank):
        for to_c in range(to_r, rank):
            from_r = padded_to_unpadded.get(to_r)
            from_c = padded_to_unpadded.get(to_c)
            if from_r is not None and from_c is not None:
                out[to_r, to_c] = unpadded_cov[from_r, from_c]
                if to_r != to_c:
                    out[to_c, to_r] = out[to_r, to_c]  # symmetrical
    return out
