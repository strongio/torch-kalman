import torch
from torch import Tensor
import numpy as np
from torch.nn import Parameter

from typing import Tuple, Optional


class Covariance(Tensor):
    @classmethod
    def from_log_cholesky(cls,
                          log_diag: Parameter,
                          off_diag: Parameter,
                          device: Optional[torch.device] = None) -> 'Covariance':
        n = len(log_diag)
        L = torch.empty(size=(n, n), device=device)
        L[np.diag_indices(n)] = torch.exp(log_diag)
        L[np.tril_indices(n=n, k=-1)] = off_diag
        L[np.triu_indices(n=n, k=1)] = 0.

        out = Covariance(size=(n, n)).to(device)
        out[:] = L.mm(L.t())
        return out

    def to_log_cholesky(self) -> Tuple[Tensor, Tensor]:
        n = self.shape[-1]
        L = torch.potrf(self)
        off_diag = L[np.tril_indices(n=n, k=-1)]
        log_diag = torch.log(torch.diag(L))
        return log_diag, off_diag

    @classmethod
    def from_std_and_corr(cls,
                          log_std_devs: Tensor,
                          corr_arctanh: Tensor,
                          device: Optional[torch.device] = None) -> 'Covariance':
        if len(log_std_devs) != 2:
            raise ValueError("This method can only be used for 2x2 covariance mats.")

        std_diag = torch.diag(torch.exp(log_std_devs)).to(device=device)
        corr_mat = torch.eye(2, device=device)
        corr_mat[0, 1] = torch.tanh(corr_arctanh)
        corr_mat[1, 0] = torch.tanh(corr_arctanh)
        return torch.mm(torch.mm(std_diag, corr_mat), std_diag)
