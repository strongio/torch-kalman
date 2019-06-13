import torch

from typing import Tuple, Optional, Sequence, Type, Iterable

from torch.nn import ParameterDict, Parameter


class Covariance(torch.Tensor):
    def __new__(cls, *args, **kwargs) -> 'Covariance':
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def from_log_cholesky(cls,
                          log_diag: torch.Tensor,
                          off_diag: torch.Tensor,
                          **kwargs) -> 'Covariance':

        assert log_diag.shape[:-1] == off_diag.shape[:-1]
        batch_dim = log_diag.shape[:-1]

        rank = log_diag.shape[-1]
        L = torch.diag_embed(torch.exp(log_diag))

        idx = 0
        for i in range(rank):
            for j in range(i):
                L[..., i, j] = off_diag[..., idx]
                idx += 1

        out = cls(size=batch_dim + (rank, rank))
        if kwargs:
            out = out.to(**kwargs)
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


class CovarianceParameterization:
    def __init__(self, rank: int):
        self.rank = rank

    @property
    def param_dict(self) -> ParameterDict:
        raise NotImplementedError

    def create(self, leading_dims: Sequence[int] = ()) -> torch.Tensor:
        raise NotImplementedError


class CovarianceFromLogCholesky(CovarianceParameterization):
    def __init__(self, rank: int):
        super().__init__(rank=rank)
        num_upper_tri = int(rank * (rank - 1) / 2)
        self._param_dict = ParameterDict()
        self._param_dict['cholesky_log_diag'] = Parameter(data=.01 * torch.randn(rank))
        self._param_dict['cholesky_off_diag'] = Parameter(data=.01 * torch.randn(num_upper_tri))

    def create(self, leading_dims: Sequence[int] = ()) -> torch.Tensor:
        cov = Covariance.from_log_cholesky(**{k.replace('cholesky_', ''): v for k, v in self.param_dict.items()})
        return cov.expand(tuple(leading_dims) + (-1, -1)).clone()

    @property
    def param_dict(self):
        return self._param_dict


class CovarianceFromStdDevs(CovarianceParameterization):
    def __init__(self, rank: int):
        super().__init__(rank=rank)
        self._param_dict = ParameterDict()
        self._param_dict['log_std_devs'] = Parameter(data=.01 * torch.randn(rank))

    def create(self, leading_dims: Sequence[int] = ()) -> torch.Tensor:
        std_devs = torch.exp(self.param_dict['log_std_devs'])
        cov = torch.diag_embed(std_devs ** 2)
        return cov.expand(tuple(leading_dims) + (-1, -1)).clone()

    @property
    def param_dict(self):
        return self._param_dict


class PartialCovariance(CovarianceParameterization):
    partial_parameterizer_cls: Type[CovarianceParameterization] = None

    def __init__(self, full_dim_names: Iterable, partial_dim_names: Iterable, diag: float = 0.0):
        self.diag = diag
        self.full_dim_names = list(full_dim_names)
        self.partial_dim_names = list(partial_dim_names)

        names_in_partial_but_not_full = set(self.partial_dim_names) - set(self.full_dim_names)
        if len(names_in_partial_but_not_full):
            raise ValueError(f"The following are present in `partial_dim_names` but not `full_dim_names`:"
                             f"\n{names_in_partial_but_not_full}")

        super().__init__(rank=len(self.full_dim_names))

        self.partial_parameterizer = self.partial_parameterizer_cls(rank=self.partial_rank)

    @property
    def partial_rank(self):
        return len(self.partial_dim_names)

    @property
    def full_rank(self):
        return len(self.full_dim_names)

    @property
    def param_dict(self) -> ParameterDict:
        return self.partial_parameterizer.param_dict

    def create(self, leading_dims: Sequence[int] = ()):
        cov = torch.eye(self.full_rank) * self.diag
        cov = cov.expand(tuple(leading_dims) + (-1, -1)).clone()

        if self.partial_rank == 0:
            return cov

        partial_cov = self.partial_parameterizer.create(leading_dims=())

        for r in range(self.partial_rank):
            for c in range(self.partial_rank):
                to_r = self.full_dim_names.index(self.partial_dim_names[r])
                to_c = self.full_dim_names.index(self.partial_dim_names[c])
                cov[..., to_r, to_c] = partial_cov[r, c]

        return cov


class PartialCovarianceFromLogCholesky(PartialCovariance):
    partial_parameterizer_cls = CovarianceFromLogCholesky
