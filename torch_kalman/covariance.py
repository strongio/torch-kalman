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
