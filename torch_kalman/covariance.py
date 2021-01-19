import math
from typing import List, Dict, Iterable, Optional, Tuple
from warnings import warn

import torch

from torch import Tensor, nn, jit
from torch_kalman.internals.utils import get_owned_kwarg


class Covariance(nn.Module):
    def __init__(self,
                 rank: int,
                 modules: Optional[nn.ModuleDict] = None,
                 time_varying_kwargs: Optional[List[str]] = None,
                 id: Optional[str] = None,
                 empty_idx: List[int] = (),
                 method: str = 'log_cholesky'):

        super(Covariance, self).__init__()

        self.id = id
        self.rank = rank

        if len(empty_idx) == 0:
            empty_idx = [self.rank + 1]  # jit doesn't seem to like empty lists
        self.empty_idx = empty_idx

        #
        self.cholesky_log_diag: Optional[nn.Parameter] = None
        self.cholesky_off_diag: Optional[nn.Parameter] = None
        self.lr_mat: Optional[nn.Parameter] = None
        self.log_std_devs: Optional[nn.Parameter] = None
        param_rank = len([i for i in range(self.rank) if i not in self.empty_idx])
        self.method = method
        if self.method == 'log_cholesky':
            self.cholesky_log_diag = nn.Parameter(.1 * torch.randn(param_rank))
            n_off = int(param_rank * (param_rank - 1) / 2)
            self.cholesky_off_diag = nn.Parameter(.1 * torch.randn(n_off))
        elif self.method == 'low_rank':
            low_rank = int(math.sqrt(param_rank))
            self.lr_mat = nn.Parameter(data=.01 * torch.randn(param_rank, low_rank))
            self.log_std_devs = nn.Parameter(data=.1 * torch.randn(param_rank) - 1)
        else:
            raise NotImplementedError(method)

        self.expected_kwargs: Optional[List[str]] = None
        self.modules = modules
        if self.modules is not None:
            assert set(time_varying_kwargs).issubset(self.modules.keys())
            self.expected_kwargs: List[str] = []
            for expected_kwarg, module in self.modules.items():
                self.expected_kwargs.append(expected_kwarg)
        self.time_varying_kwargs = time_varying_kwargs

    @jit.ignore
    def set_id(self, id: str) -> 'Covariance':
        if self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, str, Tensor]]:
        for key in (self.expected_kwargs or []):
            found_key, value = get_owned_kwarg(self.id, key, kwargs)
            key_type = 'time_varying' if key in self.time_varying_kwargs else 'static'
            yield found_key, key, key_type, value

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

    def forward(self, inputs: Dict[str, Tensor], cache: Dict[str, Tensor]) -> Tensor:
        assert self.id is not None
        key = self._get_cache_key(inputs, prefix=self.id)
        if key is not None:
            if key not in cache:
                cache[key] = self._get_padded_cov(inputs)
            cov = cache[key]
        else:
            cov = self._get_padded_cov(inputs)
        return cov

    def _get_cache_key(self, inputs: Dict[str, Tensor], prefix: str) -> Optional[str]:
        """
        Subclasses could use `inputs` to determine the cache-key
        """
        if self.time_varying_kwargs is not None:
            if len(set(inputs).intersection(self.time_varying_kwargs)) > 0:
                return None
        return f'{prefix}_static'

    def _get_padded_cov(self, inputs: Dict[str, Tensor]) -> Tensor:
        if self.method == 'log_cholesky':
            assert self.cholesky_log_diag is not None
            assert self.cholesky_off_diag is not None
            L = self.log_chol_to_chol(self.cholesky_log_diag, self.cholesky_off_diag)
            mini_cov = L @ L.t()
        elif self.method == 'low_rank':
            assert self.lr_mat is not None
            assert self.log_std_devs is not None
            mini_cov = (
                    self.lr_mat @ self.lr_mat.t() +
                    torch.diag_embed(self.log_std_devs.exp() ** 2)
            )
        else:
            raise NotImplementedError(self.method)

        if torch.isclose(mini_cov.diagonal(dim1=-2, dim2=-1), torch.zeros(1), atol=1e-12).any():
            warn(
                f"`{self.id}` has near-zero along the diagonal. Will add 1e-12 to the diagonal. "
                f"Values:\n{mini_cov.diag()}"
            )
            mini_cov = mini_cov + torch.eye(mini_cov.shape[-1]) * 1e-12

        # TODO: cache the base-cov if inputs are time-varying?
        if self.modules is not None:
            for expected_kwarg, module in self.modules.items():
                if expected_kwarg not in inputs:
                    raise TypeError(f"`{self.id}` missing required kwarg `{expected_kwarg}`")
                diag_multi = torch.diag_embed(module(inputs[expected_kwarg]))
                mini_cov = diag_multi @ mini_cov @ diag_multi

        return pad_covariance(mini_cov, [int(i not in self.empty_idx) for i in range(self.rank)])


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
