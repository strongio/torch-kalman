import torch

from typing import Tuple, Optional


class Covariance(torch.Tensor):
    def __new__(cls, *args, **kwargs) -> 'Covariance':
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def from_log_cholesky(cls,
                          log_diag: torch.Tensor,
                          off_diag: torch.Tensor,
                          device: Optional[torch.device] = None) -> 'Covariance':

        assert log_diag.shape[:-1] == off_diag.shape[:-1]
        batch_dim = log_diag.shape[:-1]

        rank = log_diag.shape[-1]
        L = torch.diag_embed(torch.exp(log_diag))

        idx = 0
        for i in range(rank):
            for j in range(i):
                L[..., i, j] = off_diag[..., idx]
                idx += 1

        out = Covariance(size=batch_dim + (rank, rank)).to(device)
        perm_shape = tuple(range(len(batch_dim))) + (-1, -2)
        out[:] = L.matmul(L.permute(perm_shape))
        return out

    def to_log_cholesky(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_dim = self.shape[:-1]
        rank = self.shape[-1]
        L = torch.cholesky(self)

        n_off = int(rank * (rank - 1) / 2)
        off_diag = torch.empty(batch_dim + (n_off,))
        idx = 0
        for i in range(rank):
            for j in range(i):
                off_diag[..., idx] = L[..., i, j]
                idx += 1
        log_diag = torch.log(torch.diagonal(L, dim1=-2, dim2=-1))
        return log_diag, off_diag


def cov_to_corr(cov: torch.Tensor) -> torch.Tensor:
    std = cov.diag().sqrt()
    return cov / std.unsqueeze(-1).matmul(std.unsqueeze(-2))
