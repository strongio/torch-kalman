import torch
import numpy as np

from typing import Tuple, Optional

from torch_kalman.utils import matrix_diag


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

        n = log_diag.shape[-1]
        L = matrix_diag(torch.exp(log_diag))

        idx = 0
        for i in range(n):
            for j in range(i):
                L[..., i, j] = off_diag[..., idx]
                idx += 1

        out = Covariance(size=batch_dim + (n, n)).to(device)
        perm_shape = tuple(range(len(batch_dim))) + (-1, -2)
        out[:] = L.matmul(L.permute(perm_shape))
        return out

    def to_log_cholesky(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.shape[-1]
        L = torch.cholesky(self)
        off_diag = L[np.tril_indices(n=n, k=-1)]
        log_diag = torch.log(torch.diag(L))
        return log_diag, off_diag
