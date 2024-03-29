import math
from typing import List, Dict, Iterable, Optional, Tuple, Sequence, Union
from warnings import warn

import torch

from torch import Tensor, nn, jit

from torch_kalman.internals.utils import get_owned_kwarg, is_near_zero
from torch_kalman.process.base import Process


def num_off_diag(rank: int) -> int:
    return int(rank * (rank - 1) / 2)


class Covariance(nn.Module):
    @classmethod
    def for_processes(cls, processes: Sequence[Process], cov_type: str, **kwargs) -> 'Covariance':
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

        if cov_type == 'initial' and (state_rank - len(no_cov_idx)) >= 10:
            # by default, use low-rank parameterization for initial cov:
            if 'method' not in kwargs:
                kwargs['method'] = 'low_rank'

        return cls(rank=state_rank, empty_idx=no_cov_idx, id=f'{cov_type}_covariance', **kwargs)

    @classmethod
    def for_measures(cls, measures: Sequence[str], **kwargs) -> 'Covariance':
        if 'method' not in kwargs and len(measures) > 5:
            kwargs['method'] = 'low_rank'
        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = 1.0
        return cls(rank=len(measures), id='measure_covariance', **kwargs)

    def __init__(self,
                 rank: int,
                 empty_idx: List[int] = (),
                 var_predict: Union[nn.Module, nn.ModuleDict] = None,
                 time_varying_kwargs: Optional[List[str]] = None,
                 id: Optional[str] = None,
                 method: str = 'log_cholesky',
                 init_diag_multi: float = 0.1,
                 var_predict_multi: float = 0.1):
        """
        :param rank: The number of elements along the diagonal.
        :param empty_idx: In some cases (e.g. process-covariance) we will
        :param var_predict: A nn.Module (or dictionary of these) that will predict a (log) multiplier for the variance.
        These should output real-values with shape `(num_groups, self.param_rank)`; these values will then be
        converted to multipliers by applying `torch.exp` (i.e. don't pass the output through a softplus). If a single
        Module is passed, you should pass `{self.id}__predictors` to the KalmanFilter's `forward` pass to supply
        predictors. If a dictionary of module(s) is passed, the key(s) of this dictionary is used to identify the
        keyword-arg that will be passed to the KalmanFilter's `forward` method (e.g. `{'group_ids' : Embedding()}`
        would allow you to call the KalmanFilter with `forward(*args, group_ids=group_ids)` and predict group-specific
        variances).
        :param time_varying_kwargs: If `var_predict_modules` is specified, we need to specify which (if any) of the
        keyword-args contain inputs that vary with each timestep.
        :param id: Identifier for this covariance. Typically left `None` and set when passed to the KalmanFilter.
        :param method: The parameterization for the covariance. The default, "log_cholesky", parameterizes the
        covariance using the cholesky factorization (which is itself split into two tensors: the log-transformed
        diagonal elements and the off-diagonal). The other currently supported option is "low_rank", which
        parameterizes the covariance with two tensors: (a) the log-transformed std-devations, and (b) a 'low rank' G*K
        tensor where G is the number of random-effects and K is int(sqrt(G)). Then the covariance is D + V @ V.t()
        where D is a diagonal-matrix with the std-deviations**2, and V is the low-rank tensor.
        :param init_diag_multi: A float that will be applied as a multiplier to the initial values along the diagonal.
        This can be useful to provide intelligent starting-values to speed up optimization.
        :param var_predict_multi: If `var_predict_modules` are standard modules like `torch.nn.Linear` or
        `torch.nn.Embedding`, the random inits can often result in somewhat unrealistic variance-multipliers; these
        poor inits can make early optimization unstable. `var_predict_multi` (default 0.1) simply multiplies the output
        of `var_predict_modules` before passing them though `torch.exp`; this serves to dampen initial outputs while
        still allowing large predictions if these are eventually warranted.
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

        self.var_predict_modules: Optional[nn.ModuleDict] = None
        if isinstance(var_predict, nn.ModuleDict):
            self.var_predict_modules = var_predict
        elif isinstance(var_predict, nn.Module):
            self.var_predict_modules = nn.ModuleDict({'predictors': var_predict})
        elif isinstance(var_predict, dict):
            self.var_predict_modules = nn.ModuleDict(var_predict)
        elif var_predict is not None:
            raise ValueError(
                f"If `var_predict` is passed, should be a nn.Module or dictionary of these. Got {type(var_predict)}"
            )

        self.expected_kwargs: Optional[List[str]] = None
        self.time_varying_kwargs = time_varying_kwargs

        self.var_predict_multi = var_predict_multi
        if self.var_predict_modules is not None:
            if time_varying_kwargs is not None:
                assert set(time_varying_kwargs).issubset(self.var_predict_modules.keys())
            self.expected_kwargs: List[str] = []
            for expected_kwarg_plus, module in self.var_predict_modules.items():
                pname, _, expected_kwarg = expected_kwarg_plus.rpartition("__")
                self.expected_kwargs.append(expected_kwarg)

    @jit.ignore
    def set_id(self, id: str) -> 'Covariance':
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, str, Tensor]]:
        for key in (self.expected_kwargs or []):
            found_key, value = get_owned_kwarg(self.id, key, kwargs)
            key_type = 'time_varying' if key in (self.time_varying_kwargs or []) else 'static'
            yield found_key, key, key_type, torch.as_tensor(value)

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
        params: Dict[str, List[Tensor]] = {}
        if self.var_predict_modules is not None:
            for expected_kwarg_plus, module in self.var_predict_modules.items():
                pname, _, expected_kwarg = expected_kwarg_plus.rpartition("__")
                if expected_kwarg not in inputs:
                    raise TypeError(f"`{self.id}` missing required kwarg `{expected_kwarg}`")
                if pname == '':
                    pname = 'var_multi'
                if pname not in params:
                    params[pname] = []
                params[pname].append(self.var_predict_multi * module(inputs[expected_kwarg]))

        if self.method == 'log_cholesky':
            assert self.cholesky_log_diag is not None
            assert self.cholesky_off_diag is not None
            cholesky_log_diag = self.cholesky_log_diag
            cholesky_off_diag = self.cholesky_off_diag
            if 'cholesky_off_diag' in params and 'cholesky_log_diag' in params:
                cholesky_log_diag = cholesky_log_diag + torch.sum(torch.stack(params['cholesky_log_diag'], 0))
                cholesky_off_diag = cholesky_off_diag + torch.sum(torch.stack(params['cholesky_off_diag'], 0))
            else:
                assert 'cholesky_off_diag' not in params and 'cholesky_log_diag' not in params

            L = self.log_chol_to_chol(cholesky_log_diag, cholesky_off_diag)
            mini_cov = L @ L.t()
        elif self.method == 'low_rank':
            if len(params) > 0 and (list(params.keys()) != ['var_multi']):
                # TODO
                raise NotImplementedError
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
            mini_cov = mini_cov + torch.eye(mini_cov.shape[-1]) * 1e-12

        if 'var_multi' in params.keys():
            diag_multi = torch.exp(torch.sum(torch.stack(params['var_multi'], 0), 0))
            diag_multi = torch.diag_embed(diag_multi)
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

    out = torch.zeros(unpadded_cov.shape[:-2] + (rank, rank))
    for to_r in range(rank):
        for to_c in range(to_r, rank):
            from_r = padded_to_unpadded.get(to_r)
            from_c = padded_to_unpadded.get(to_c)
            if from_r is not None and from_c is not None:
                out[..., to_r, to_c] = unpadded_cov[..., from_r, from_c]
                if to_r != to_c:
                    out[..., to_c, to_r] = out[..., to_r, to_c]  # symmetrical
    return out
