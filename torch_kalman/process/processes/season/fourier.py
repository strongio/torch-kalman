from typing import Generator, Tuple, Optional, Union, Sequence, Dict, List

from torch import Tensor
from torch.nn import Parameter, ParameterDict

from torch_kalman.process import Process

import numpy as np

from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.process.utils.dt_tracker import DTTracker
from torch_kalman.process.utils.fourier import fourier_tensor


class FourierSeason(Process):
    def __init__(self,
                 id: str,
                 seasonal_period: Union[int, float],
                 K: Union[int, float],
                 decay: Union[bool, Tuple[float, float]] = False,
                 season_start: Union[str, None, bool] = None,
                 dt_unit: Optional[str] = None):

        # handle datetimes:
        self.dt_tracker = DTTracker(season_start=season_start, dt_unit=dt_unit, process_id=id)

        # season structure:
        self.seasonal_period = seasonal_period
        if isinstance(K, float):
            assert K.is_integer()
        self.K = int(K)

        self.decay: Optional[Bounded] = None
        if decay:
            assert not isinstance(decay, bool), "decay should be floats of bounds (or False for no decay)"
            assert decay[0] > 0. and decay[1] <= 1.0
            self.decay = Bounded(*decay)

        state_elements, list_of_trans_kwargs = self._setup(decay=decay)
        super().__init__(id=id, state_elements=state_elements)
        for trans_kwargs in list_of_trans_kwargs:
            self._set_transition(**trans_kwargs)

    def _setup(self, decay: bool) -> Tuple[List[str], List[Dict]]:
        state_elements = []
        transitions = []
        for r in range(self.K):
            for c in range(2):
                element_name = f"{r},{c}"
                state_elements.append(element_name)
                trans_kwargs = {'from_element': element_name,
                                'to_element': element_name,
                                'value': self.decay.get_value if decay else 1.0}
                transitions.append(trans_kwargs)

        return state_elements, transitions

    def param_dict(self) -> ParameterDict:
        p = ParameterDict()
        if self.decay is not None:
            p['decay'] = self.decay.parameter
        return p

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        raise NotImplementedError

    def add_measure(self, measure: str) -> 'FourierSeason':
        raise NotImplementedError


class FourierSeasonDynamic(FourierSeason):
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
                  start_datetimes: Optional[np.ndarray] = None) -> ProcessForBatch:

        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps)

        # determine the delta (integer time accounting for different groups having different start datetimes)
        delta = self.dt_tracker.get_delta(for_batch.num_groups, for_batch.num_timesteps, start_datetimes=start_datetimes)

        # determine season:
        season = delta % self.seasonal_period

        # generate the fourier tensor:
        fourier_tens = fourier_tensor(time=Tensor(season), seasonal_period=self.seasonal_period, K=self.K)

        for state_element in self.state_elements:
            if state_element == 'position':
                continue
            r, c = (int(x) for x in state_element.split(sep=","))
            for_batch.adjust_transition(from_element=state_element,
                                        to_element='position',
                                        adjustment=fourier_tens[:, :, r, c])

        return for_batch

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements[:-1]

    def add_measure(self, measure: str) -> 'FourierSeasonDynamic':
        self._set_measure(measure=measure, state_element='position', value=1.0)
        return self


class FourierSeasonFixed(FourierSeason):
    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  start_datetimes: Optional[np.ndarray] = None) -> ProcessForBatch:

        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps)

        # determine the delta (integer time accounting for different groups having different start datetimes)
        delta = self.dt_tracker.get_delta(for_batch.num_groups, for_batch.num_timesteps, start_datetimes=start_datetimes)

        # determine season:
        season = delta % self.seasonal_period

        # generate the fourier tensor:
        fourier_tens = fourier_tensor(time=Tensor(season), seasonal_period=self.seasonal_period, K=self.K)

        for measure in self.measures:
            for state_element in self.state_elements:
                r, c = (int(x) for x in state_element.split(sep=","))
                for_batch.adjust_measure(measure=measure, state_element=state_element, adjustment=fourier_tens[:, :, r, c])

        return for_batch

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return []

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> 'FourierSeasonFixed':
        for state_element in self.state_elements:
            self._set_measure(measure=measure, state_element=state_element, value=0.)
        return self


# noinspection SpellCheckingInspection
class TBATS(FourierSeason):
    """
    This implementation is not complete: (1) does not allow decay, (2) does not offset seasons according to group start date.
    """

    def __init__(self,
                 id: str,
                 seasonal_period: Union[int, float],
                 K: Union[int, float],
                 decay: Union[bool, Tuple[float, float]] = False,
                 season_start: Union[str, None, bool] = None,
                 dt_unit: Optional[str] = None):
        super().__init__(id=id,
                         seasonal_period=seasonal_period,
                         K=K,
                         decay=decay,
                         season_start=season_start,
                         dt_unit=dt_unit)
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

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> 'TBATS':
        for se in self.measured_state_elements:
            self._set_measure(measure=measure, state_element=se, value=1.0)
        return self

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements
