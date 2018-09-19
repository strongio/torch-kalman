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

    @classmethod
    def from_diag(cls, diag: Parameter) -> Tensor:
        n = len(diag)
        out = Covariance(size=(n, n))
        out[np.diag_indices(n)] = diag
        out[np.tril_indices(n=n)] = 0.
        out[np.triu_indices(n=n)] = 0.
        return out
