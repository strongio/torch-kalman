from typing import Tuple, Optional, Union, Sequence, Dict, List, Callable

from torch import Tensor
from torch.nn import ParameterDict, Module

from torch_kalman.process import Process

import numpy as np

from torch_kalman.process.utils.bounded import Bounded

from torch_kalman.process.utils.fourier import fourier_tensor
from torch_kalman.internals.utils import split_flat
from torch_kalman.utils.datetime import DateTimeHelper


class _FourierSeason(Process):
    def __init__(self,
                 id: str,
                 seasonal_period: Union[int, float],
                 K: Union[int, float],
                 decay: Union[bool, Tuple[float, float]] = False,
                 dt_unit: Optional[str] = None,
                 initial_state: Optional[torch.nn.Module] = None):

        # season structure:
        self.seasonal_period = seasonal_period
        if isinstance(K, float):
            assert K.is_integer()
        self.K = int(K)

        self.decay = None
        if decay:
            assert decay[0] > 0. and decay[1] <= 1.0
            self.decay = Bounded(*decay)

        state_elements, list_of_trans_kwargs = self._setup(decay=decay)

        super().__init__(id=id, state_elements=state_elements, initial_state=initial_state)

        self._dt_helper = DateTimeHelper(dt_unit=dt_unit)

        for trans_kwargs in list_of_trans_kwargs:
            self._set_transition(**trans_kwargs)

    def _setup(self, decay: bool) -> Tuple[List[str], List[Dict]]:
        state_elements = []
        transitions = []
        for r in range(self.K):
            for c in range(2):
                element_name = f"{r},{c}"
                state_elements.append(element_name)
                trans_kwargs = {
                    'from_element': element_name,
                    'to_element': element_name,
                    'value': self.decay.get_value if decay else 1.0
                }
                transitions.append(trans_kwargs)

        return state_elements, transitions

    def param_dict(self) -> ParameterDict:
        p = super().param_dict()
        if self.decay is not None:
            p['decay'] = self.decay.parameter
        return p

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        raise NotImplementedError

    def add_measure(self, measure: str) -> '_FourierSeason':
        raise NotImplementedError


class FourierSeason(_FourierSeason):
    """
    Process representing seasons using a fourier-series.
    """

    def __init__(self,
                 id: str,
                 seasonal_period: Union[int, float],
                 K: Union[int, float],
                 fixed: bool = False,
                 decay: Union[bool, Tuple[float, float]] = False,
                 dt_unit: Optional[str] = None,
                 initial_state: Optional[Module] = None):

        """
        :param id: Unique name for this instance.
        :param seasonal_period: The seasonal period (e.g. 24 for daily season in hourly data, 365.25 for yearly season
        in daily data)
        :param K: The "K" parameter of the fourier series, see `fourier_tensor`.
        :param decay: Optional (float,float) boundaries for decay (between 0 and 1). Analogous to dampening a trend --
        the state will revert to zero as we get further from the last observation. This can be useful if two processes
        are capturing the same seasonal pattern: one can be more flexible, but with decay have a tendency to revert to
        zero, while the other is less variable but extrapolates into the future.
        :param dt_unit: Currently supports {'Y', 'D', 'h', 'm', 's'}. 'W' is experimentally supported.
        :param initial_state: Optional, a callable (typically a torch.nn.Module). When the KalmanFilter is called,
        keyword-arguments can be passed to initial_state in the format `{this_process}_initial_state__{kwarg}`.
        """
        self.fixed = fixed
        super().__init__(
            id=id, seasonal_period=seasonal_period, K=K, decay=decay, dt_unit=dt_unit, initial_state=initial_state
        )

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  start_datetimes: Optional[np.ndarray] = None):

        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps)

        # determine the delta (integer time accounting for different groups having different start datetimes)
        if start_datetimes is None:
            if self._dt_helper.dt_unit:
                raise TypeError("Missing argument `start_datetimes`.")
            start_datetimes = np.zeros(num_groups)
        delta = self._dt_helper.make_delta_grid(start_datetimes, num_timesteps)

        # determine season:
        season = delta % self.seasonal_period

        # generate the fourier tensor:
        fourier_tens = fourier_tensor(time=Tensor(season), seasonal_period=self.seasonal_period, K=self.K)

        for measure in self.measures:
            for state_element in self.state_elements:
                r, c = (int(x) for x in state_element.split(sep=","))
                for_batch._adjust_measure(
                    measure=measure,
                    state_element=state_element,
                    adjustment=split_flat(fourier_tens[:, :, r, c], dim=1)
                )

        return for_batch

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return [] if self.fixed else self.state_elements

    def add_measure(self, measure: str) -> 'FourierSeasonFixed':
        for state_element in self.state_elements:
            self._set_measure(measure=measure, state_element=state_element, value=0.)
        return self


class FourierSeason2(_FourierSeason):
    def _setup(self, decay: bool) -> Tuple[List[str], List[Dict]]:
        state_elements, transitions = super()._setup(decay=decay)

        # all fourier components transition into position:
        for se in state_elements:
            transitions.append({'from_element': se, 'to_element': 'position', 'value': 0.})

        # add position. note that it doesn't transition into itself
        state_elements.append('position')
        return state_elements, transitions

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  start_datetimes: Optional[np.ndarray] = None):

        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps)

        # determine the delta (integer time accounting for different groups having different start datetimes)
        if start_datetimes is None:
            start_datetimes = np.zeros(num_groups)
        delta = self._dt_helper.make_delta_grid(start_datetimes, num_timesteps)

        # determine season:
        season = delta % self.seasonal_period

        # generate the fourier tensor:
        fourier_tens = fourier_tensor(time=Tensor(season), seasonal_period=self.seasonal_period, K=self.K)

        for state_element in self.state_elements:
            if state_element == 'position':
                continue
            r, c = (int(x) for x in state_element.split(sep=","))
            for_batch._adjust_transition(
                from_element=state_element,
                to_element='position',
                adjustment=split_flat(fourier_tens[:, :, r, c], dim=1)
            )

        return for_batch

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements[:-1]

    def add_measure(self, measure: str) -> 'FourierSeason2':
        self._set_measure(measure=measure, state_element='position', value=1.0)
        return self


class TBATS(_FourierSeason):
    """
    This implementation is not complete: (1) does not allow decay, (2) does not offset seasons according to group start
    date.
    """

    def __init__(self,
                 id: str,
                 seasonal_period: Union[int, float],
                 K: Union[int, float],
                 decay: Union[bool, Tuple[float, float]] = False,
                 dt_unit: Optional[str] = None):

        if decay:
            raise NotImplementedError(f"{type(self).__name__} does not yet support decay.")
        if dt_unit:
            raise NotImplementedError(f"{type(self).__name__} does not yet support datetimes.")

        super().__init__(id=id,
                         seasonal_period=seasonal_period,
                         K=K,
                         decay=decay)
        self.measured_state_elements = [se for se in self.state_elements if '*' not in se]

    def _setup(self, decay: bool) -> Tuple[List[str], List[Dict]]:
        if decay:
            raise NotImplementedError

        state_elements = []
        transitions = []
        for j in range(self.K):
            sj = f"s{j}"
            s_star_j = f"s*{j}"
            state_elements.extend([sj, s_star_j])
            transitions.extend([
                {'to_element': sj, 'from_element': sj, 'value': np.cos(self.lam(j))},
                {'to_element': sj, 'from_element': s_star_j, 'value': np.sin(self.lam(j))},
                {'to_element': s_star_j, 'from_element': sj, 'value': -np.sin(self.lam(j))},
                {'to_element': s_star_j, 'from_element': s_star_j, 'value': np.cos(self.lam(j))}
            ])

        return state_elements, transitions

    def lam(self, j: int):
        return 2. * np.pi * j / self.seasonal_period

    def add_measure(self, measure: str) -> 'TBATS':
        for se in self.measured_state_elements:
            self._set_measure(measure=measure, state_element=se, value=1.0)
        return self

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements
