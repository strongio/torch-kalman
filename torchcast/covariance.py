import math
from typing import List, Dict, Iterable, Optional, Tuple, Sequence, Union
from warnings import warn

import torch

from torch import Tensor, nn, jit
from typing_extensions import Final

from torchcast.internals.utils import get_owned_kwarg, is_near_zero
from torchcast.process.base import Process


def num_off_diag(rank: int) -> int:
    return int(rank * (rank - 1) / 2)


class Covariance(nn.Module):
    """
    The :class:`.Covariance` object can be used in setting where you'd like to predict the variance in your forecasts.
    For example, if you're training on diverse time-serieses that vary in scale/behavior, you could use an
    :class:`torch.nn.Embedding` with the group-ids as predictors:

    .. code-block:: python3

        kf = KalmanFilter(
            measures=measures,
            processes=processes,
            measure_covariance=Covariance.for_measures(
                measures,
                predict_variance=torch.nn.Embedding(len(group_ids), len(measures), padding_idx=0)
            ),
            process_covariance=Covariance.for_processes(
                processes,
                predict_variance=torch.nn.Embedding(
                    len(group_ids),
                    Covariance.for_processes(processes).param_rank,
                    padding_idx=0
                )
            )
        )

    """
    var_predict_multi: Final[float] = 0.1
    """
    :cvar var_predict_multi: If ``predict_variance`` are standard modules like :class:`torch.nn.Linear` or
     :class:`torch.nn.Embedding`, the random inits can often result in extreme variance-multipliers; these poor
     inits can make early optimization unstable. `var_predict_multi` (default 0.1) simply multiplies the output
     of ``predict_variance`` before passing them though :func:`torch.exp`; this serves to dampen initial outputs
     while still allowing large predictions if these are eventually warranted.
    """

    @classmethod
    def for_processes(cls,
                      processes: Sequence[Process],
                      cov_type: str = 'process',
                      predict_variance: Optional[nn.Module] = None,
                      **kwargs) -> 'Covariance':
        """
        :param processes: The ``processes`` used in your :class:`.KalmanFilter`.
        :param cov_type: The type of covariance, either 'process' or 'initial' (default: 'process').
        :param predict_variance: A :class:`torch.nn.Module` that will predict a (log) multiplier for the variance.
         These should output real-values with shape ``(num_groups, self.param_rank)``; these values will then be
         converted to multipliers by applying :func:`torch.exp` (i.e. don't pass the output through a softplus).
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        :return: A :class:`.Covariance` object that can be used in your :class:`.KalmanFilter`.
        """
        assert cov_type in {'process', 'initial'}
        state_rank = 0
        no_cov_idx = []
        for p in processes:
            no_cov_elements = getattr(p, f'no_{cov_type[0]}cov_state_elements') or []
            for i, se in enumerate(p.state_elements):
                if se in no_cov_elements:
                    no_cov_idx.append(state_rank + i)
            state_rank += len(p.state_elements)

        if cov_type == 'process':
            # by default, assume process cov is less than measure cov:
            if 'init_diag_multi' not in kwargs:
                kwargs['init_diag_multi'] = .01
        elif cov_type == 'initial':
            if (state_rank - len(no_cov_idx)) >= 10:
                # by default, use low-rank parameterization for initial cov:
                if 'method' not in kwargs:
                    kwargs['method'] = 'low_rank'
        else:
            raise ValueError(f"Unrecognized cov_type {cov_type}, expected 'initial' or 'process'.")

        return cls(
            rank=state_rank,
            empty_idx=no_cov_idx,
            id=f'{cov_type}_covariance',
            predict_variance=predict_variance,
            **kwargs
        )

    @classmethod
    def for_measures(cls,
                     measures: Sequence[str],
                     predict_variance: Optional[nn.Module] = None,
                     **kwargs) -> 'Covariance':
        """
        :param measures: The ``measures`` used in your :class:`.KalmanFilter`.
        :param predict_variance: A :class:`torch.nn.Module` that will predict a (log) multiplier for the variance.
         These should output real-values with shape ``(num_groups, num_measures)``; these values will then be
         converted to multipliers by applying :func:`torch.exp` (i.e. don't pass the output through a softplus).
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        :return: A :class:`.Covariance` object that can be used in your :class:`.KalmanFilter`.
        """
        if isinstance(measures, str):
            measures = [measures]
            warn(f"`measures` should be a list of strings not a string; interpreted as `{measures}`.")
        if 'method' not in kwargs and len(measures) > 5:
            kwargs['method'] = 'low_rank'
        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = 1.0
        return cls(rank=len(measures), id='measure_covariance', predict_variance=predict_variance, **kwargs)

    def __init__(self,
                 rank: int,
                 empty_idx: List[int] = (),
                 predict_variance: Optional[nn.Module] = None,
                 id: Optional[str] = None,
                 method: str = 'log_cholesky',
                 init_diag_multi: float = 0.1):
        """
        You should rarely call this directly. Instead, call :func:`Covariance.for_measures` and
        :func:`Covariance.for_processes`.

        :param rank: The number of elements along the diagonal.
        :param empty_idx: In some cases (e.g. process-covariance) we will have some elements with no variance.
        :param predict_variance: A :class:`torch.nn.Module` that will predict a (log) multiplier for the variance.
         These should output real-values with shape ``(num_groups, self.param_rank)``; these values will then be
         converted to multipliers by applying :func:`torch.exp` (i.e. don't pass the output through a softplus).
        :param id: Identifier for this covariance. Typically left ``None`` and set when passed to the
         :class:`.KalmanFilter`.
        :param method: The parameterization for the covariance. The default, "log_cholesky", parameterizes the
         covariance using the cholesky factorization (which is itself split into two tensors: the log-transformed
         diagonal elements and the off-diagonal). The other currently supported option is "low_rank", which
         parameterizes the covariance with two tensors: (a) the log-transformed std-devations, and (b) a 'low rank' G*K
         tensor where G is the number of random-effects and K is int(sqrt(G)). Then the covariance is D + V @ V.t()
         where D is a diagonal-matrix with the std-deviations**2, and V is the low-rank tensor.
        :param init_diag_multi: A float that will be applied as a multiplier to the initial values along the diagonal.
         This can be useful to provide intelligent starting-values to speed up optimization.
        """

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
        self.param_rank = len([i for i in range(self.rank) if i not in self.empty_idx])
        self.method = method
        if self.method == 'log_cholesky':
            self.cholesky_log_diag = nn.Parameter(.1 * torch.randn(self.param_rank) + math.log(init_diag_multi))
            self.cholesky_off_diag = nn.Parameter(.1 * torch.randn(num_off_diag(self.param_rank)))
        elif self.method == 'low_rank':
            low_rank = int(math.sqrt(self.param_rank))
            self.lr_mat = nn.Parameter(data=.01 * torch.randn(self.param_rank, low_rank))
            self.log_std_devs = nn.Parameter(data=.1 * torch.randn(self.param_rank) + math.log(init_diag_multi))
        else:
            raise NotImplementedError(method)

        self.var_predict_module = predict_variance

        self.expected_kwarg = '' if self.var_predict_module is None else 'X'

    @jit.ignore
    def set_id(self, id: str) -> 'Covariance':
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, Tensor]]:
        if self.expected_kwarg:
            found_key, value = get_owned_kwarg(self.id, self.expected_kwarg, kwargs)
            yield found_key, self.expected_kwarg, torch.as_tensor(value)

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

    def _get_mini_cov(self) -> Tensor:
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

        if is_near_zero(mini_cov.diagonal(dim1=-2, dim2=-1), atol=1e-12).any():
            warn(
                f"`{self.id}` has near-zero along the diagonal. Will add 1e-12 to the diagonal. "
                f"Values:\n{mini_cov.diag()}"
            )
            mini_cov = mini_cov + torch.eye(mini_cov.shape[-1], device=mini_cov.device, dtype=mini_cov.dtype) * 1e-12
        return mini_cov

    def forward(self, input: Optional[Tensor] = None, _ignore_input: bool = False) -> Tensor:
        mini_cov = self._get_mini_cov()

        pred = None
        if self.var_predict_module is not None and not _ignore_input:
            pred = self.var_predict_multi * self.var_predict_module(input)
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                raise RuntimeError(f"{self.id}'s `predict_variance` produced nans/infs")
            if len(pred.shape) == 1:
                raise ValueError(
                    f"{self.id} `predict_variance` module output should have 2D output, got {len(pred.shape)}"
                )
            elif pred.shape[-1] not in (1, self.param_rank):
                raise ValueError(
                    f"{self.id} `predict_variance` module output should have `shape[-1]` of "
                    f"{self.param_rank}, got {pred.shape[-1]}"
                )
        if pred is not None:
            diag_multi = torch.diag_embed(torch.exp(pred))
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

    out = torch.zeros(unpadded_cov.shape[:-2] + (rank, rank), device=unpadded_cov.device, dtype=unpadded_cov.dtype)
    for to_r in range(rank):
        for to_c in range(to_r, rank):
            from_r = padded_to_unpadded.get(to_r)
            from_c = padded_to_unpadded.get(to_c)
            if from_r is not None and from_c is not None:
                out[..., to_r, to_c] = unpadded_cov[..., from_r, from_c]
                if to_r != to_c:
                    out[..., to_c, to_r] = out[..., to_r, to_c]  # symmetrical
    return out
