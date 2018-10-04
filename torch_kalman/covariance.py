import torch
from torch import Tensor
import numpy as np
from torch.nn import Parameter


# noinspection PyAbstractClass
class Covariance(Tensor):
    @classmethod
    def from_log_cholesky(cls, log_diag: Parameter, off_diag: Parameter) -> Tensor:
        n = len(log_diag)
        L = Tensor(size=(n, n))
        L[np.diag_indices(n)] = torch.exp(log_diag)
        L[np.tril_indices(n=n, k=-1)] = off_diag
        L[np.triu_indices(n=n, k=1)] = 0.
        return L.mm(L.t())

    def to_log_cholesky(self):
        n = self.shape[-1]
        L = torch.potrf(self)
        off_diag = L[np.tril_indices(n=n, k=-1)]
        log_diag = torch.log(torch.diag(L))
        return log_diag, off_diag

    @classmethod
    def from_std_and_corr(cls, log_std_devs, corr_sigmoid):
        if len(log_std_devs) != 2:
            raise ValueError("This method can only be used for 2x2 covariance mats.")

        std_diag = torch.diag(torch.exp(log_std_devs))
        corr_mat = torch.eye(2)
        corr_mat[0, 1] = torch.sigmoid(corr_sigmoid)
        corr_mat[1, 0] = torch.sigmoid(corr_sigmoid)
        return torch.mm(torch.mm(std_diag, corr_mat), std_diag)
