import copy
import itertools
import math
from typing import Optional, Tuple, Iterable, Dict, Sequence, Union
from warnings import warn

import numpy as np

import torch
from torch import jit, nn, Tensor
from torch_kalman.internals.utils import zpad

from torch_kalman.process.base import Process
from torch_kalman.process.regression import _RegressionBase
from torch_kalman.process.utils import SingleOutput, Multi, Bounded, TimesToFourier


class _Season:

    @staticmethod
    def _get_dt_unit_ns(dt_unit_str: str) -> int:
        dt_unit = np.timedelta64(1, dt_unit_str)
        dt_unit_ns = dt_unit / np.timedelta64(1, 'ns')
        dt_unit_ns.is_integer()
        return int(dt_unit_ns)

    @jit.ignore
    def _get_offsets(self, start_datetimes: np.ndarray) -> np.ndarray:
        ns_since_epoch = (start_datetimes.astype("datetime64[ns]") - np.datetime64(0, 'ns')).view('int64')
        offsets = ns_since_epoch % (self.period * self.dt_unit_ns) / self.dt_unit_ns
        return torch.as_tensor(offsets.astype('float32')).view(-1, 1, 1)

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, str, Tensor]]:
        if self.dt_unit_ns is None:
            offsets = torch.zeros(1)
        else:
            offsets = self._get_offsets(kwargs['start_datetimes'])
        kwargs['current_times'] = offsets + kwargs['current_timestep']
        for found_key, key_name, key_type, value in Process.get_kwargs(self, kwargs):
            if found_key == 'current_times' and self.dt_unit_ns is not None:
                found_key = 'start_datetimes'
            yield found_key, key_name, key_type, value


class FourierSeason(_Season, _RegressionBase):
    """
    A process which captures seasonal patterns using fourier serieses. Essentially a `LinearModel` where the
    model-matrix construction is done for you. Best suited for static seasonal patterns; for evolving ones `TBATS` is
    recommended.
    """

    def __init__(self,
                 id: str,
                 dt_unit: Optional[str],
                 period: float,
                 K: int,
                 measure: Optional[str] = None,
                 process_variance: bool = False,
                 decay: Optional[Union[nn.Module, Tuple[float, float]]] = None):
        """
        :param id: A unique identifier for this process
        :param dt_unit: A string indicating the time-units used in the kalman-filter -- i.e., how far we advance with
        every timestep. Passed to `numpy.timedelta64(1, dt_unit)`.
        :param period: The number of `dt_units` it takes to get through a full season. Does not have to be an integer
        (e.g. 365.25 for yearly season on daily-data).
        :param K: The number of the fourier components
        :param measure: The name of the measure for this process.
        :param process_variance: TODO
        :param decay: TODO
        """

        self.dt_unit_ns: Optional[int] = None if dt_unit is None else self._get_dt_unit_ns(dt_unit)
        self.period = period

        if isinstance(decay, tuple) and (decay[0] ** period) < .01:
            warn(
                f"Given the seasonal period, the lower bound on `{id}`'s `decay` ({decay}) may be too low to "
                f"generate useful gradient information for optimization."
            )

        state_elements = []
        for j in range(K):
            state_elements.append(f'sin{j}')
            state_elements.append(f'cos{j}')

        super().__init__(
            id=id,
            predictors=state_elements,
            measure=measure,
            h_module=TimesToFourier(K=K, seasonal_period=float(period)),
            process_variance=process_variance,
            decay=decay
        )
        self.h_kwarg = 'current_times'
        assert len(self.time_varying_kwargs) == 1
        self.time_varying_kwargs[0] = 'current_times'


class TBATS(_Season, Process):
    """
    Named after the paper from De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011); in that paper TBATS refers to
    the whole model; here it labels the novel approach to modeling seasonality that they proposed.
    """

    def __init__(self,
                 id: str,
                 period: float,
                 dt_unit: Optional[str],
                 K: int,
                 measure: Optional[str] = None,
                 process_variance: bool = True,
                 decay: Optional[Union[nn.ModuleDict, Tuple[float, float]]] = None,
                 decay_kwarg: Optional[str] = None):
        self.period = float(period)
        self.dt_unit_ns = None if dt_unit is None else self._get_dt_unit_ns(dt_unit)

        if decay_kwarg is None:
            # assert not isinstance(decay, nn.Module) # TODO
            decay_kwarg = ''

        if isinstance(decay, tuple) and (decay[0] ** period) < .01:
            warn(
                f"Given the seasonal period, the lower bound on `{id}`'s `decay` ({decay}) may be too low to "
                f"generate useful gradient information for optimization."
            )

        state_elements, transitions, h_tensor = self._setup(K=K, period=period, decay=decay)

        init_mean_kwargs = ['start_offsets']
        if decay_kwarg:
            init_mean_kwargs.append(decay_kwarg)
        super(TBATS, self).__init__(
            id=id,
            state_elements=state_elements,
            f_tensors=transitions if decay is None else None,
            f_modules=transitions if decay is not None else None,
            h_tensor=torch.tensor(h_tensor),
            measure=measure,
            no_pcov_state_elements=[] if process_variance else state_elements,
            init_mean_kwargs=init_mean_kwargs,
            f_kwarg=decay_kwarg
        )

    def _setup(self,
               K: int,
               period: float,
               decay: Optional[Union[nn.Module, Tuple[float, float]]]) -> Tuple[Sequence[str], dict, Sequence[float]]:

        if isinstance(decay, nn.Module):
            decay = [copy.deepcopy(decay) for _ in range(K * 2)]
        if isinstance(decay, (list, tuple)):
            are_modules = [isinstance(m, nn.Module) for m in decay]
            if any(are_modules):
                assert all(are_modules), "`decay` is a list with some modules on some other types"
                assert len(decay) == K * 2, "`decay` is a list of modules, but its length != K*2"
            else:
                assert len(decay) == 2, "if `decay` is not list of modules, should be (float,float)"
                decay = [SingleOutput(transform=Bounded(*decay)) for _ in range(K * 2)]

        state_elements = []
        f_tensors = {}
        f_modules = {}
        h_tensor = []
        for j in range(1, K + 1):
            sj = f"s{j}"
            state_elements.append(sj)
            h_tensor.append(1.)
            s_star_j = f"s*{j}"
            state_elements.append(s_star_j)
            h_tensor.append(0.)

            lam = torch.tensor(2. * math.pi * j / period)

            f_tensors[f'{sj}->{sj}'] = torch.cos(lam)
            f_tensors[f'{sj}->{s_star_j}'] = -torch.sin(lam)
            f_tensors[f'{s_star_j}->{sj}'] = torch.sin(lam)
            f_tensors[f'{s_star_j}->{s_star_j}'] = torch.cos(lam)

            if decay:
                for from_, to_ in itertools.product([sj, s_star_j], [sj, s_star_j]):
                    tkey = f'{from_}->{to_}'
                    which = 2 * (j - 1) + int(from_ == to_)
                    f_modules[tkey] = nn.Sequential(decay[which], Multi(f_tensors[tkey]))

        if f_modules:
            assert len(f_modules) == len(f_tensors)
            return state_elements, f_modules, h_tensor
        else:
            return state_elements, f_tensors, h_tensor

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, str, Tensor]]:
        if self.dt_unit_ns is None:
            offsets = torch.zeros(1)
        else:
            offsets = self._get_offsets(kwargs['start_datetimes'])
        kwargs['start_offsets'] = offsets
        for found_key, key_name, key_type, value in Process.get_kwargs(self, kwargs):
            if found_key == 'start_offsets' and self.dt_unit_ns is not None:
                found_key = 'start_datetimes'
            yield found_key, key_name, key_type, value

    def get_initial_state_mean(self, input: Optional[Dict[str, Tensor]] = None) -> Tensor:
        assert input is not None
        if self.dt_unit_ns is None:
            return self.init_mean
        start_offsets = input['start_offsets']
        num_groups = start_offsets.shape[0]

        if self.f_kwarg == '':
            f_input = torch.empty(0)  # TODO: support `None`
        else:
            f_input = input[self.f_kwarg]

        F = self.f_forward(f_input)
        if F.shape[0] != num_groups:
            F = F.expand(num_groups, -1, -1)

        if abs(float(int(self.period)) - float(self.period)) > .00001:
            # TODO: does jit have `int.is_integer()` method?
            raise NotImplementedError

        means = []
        mean = self.init_mean.expand(num_groups, -1).unsqueeze(-1)
        for i in range(int(self.period)):
            means.append(mean.squeeze(-1))
            mean = F @ mean

        groups = [i for i in range(num_groups)]
        times = [int(start_offsets[i].item()) for i in groups]
        return torch.stack(means, 1)[(groups, times)]


class DiscreteSeason(Process):
    """
    TODO
    """

    def __init__(self,
                 id: str,
                 num_seasons: int,
                 season_duration: int = 1,
                 measure: Optional[str] = None,
                 process_variance: bool = False,
                 decay: Optional[Tuple[float, float]] = None):
        if isinstance(decay, tuple) and (decay[0] ** (num_seasons * season_duration)) < .01:
            warn(
                f"Given the seasonal period, the lower bound on `{self.id}`'s `decay` ({decay}) may be too low to "
                f"generate useful gradient information for optimization."
            )

        f_modules = self._make_f_modules(num_seasons, season_duration, decay)
        state_elements = [zpad(i, n=len(str(num_seasons))) for i in range(num_seasons)]
        super(DiscreteSeason, self).__init__(
            id=id,
            state_elements=state_elements,
            measure=measure,
            h_tensor=torch.tensor([1.] + [0.] * (num_seasons - 1)),
            f_modules=f_modules,
            f_kwarg='current_timestep',
            init_mean_kwargs=['start_datetimes'],
            time_varying_kwargs=['current_timestep'],
            no_pcov_state_elements=[] if process_variance else state_elements
        )

    def _make_f_modules(self,
                        num_seasons: int,
                        season_duration: int,
                        decay: Optional[Tuple[float, float]]) -> nn.ModuleDict:
        raise NotImplementedError
