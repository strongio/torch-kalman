from typing import List, Dict

import torch

from torch import Tensor, nn


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

        self.cache: Dict[str, Tensor] = {'null': torch.empty(0)}  # jit doesn't like empty

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

    def forward(self, input: Tensor) -> Tensor:
        if self.method == 'log_cholesky':
            num_groups = input.shape[0]
            key = 'cov'
            if key not in self.cache:
                L = self.log_chol_to_chol(self.cholesky_log_diag, self.cholesky_off_diag)
                self.cache[key] = L @ L.t()
            cov = self.cache[key]
            # TODO: predicting diag-multi?
            return cov.expand(num_groups, -1, -1)
        else:
            raise NotImplementedError(self.method)
