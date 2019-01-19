from typing import Generator, Tuple, Optional, Union, Sequence, Dict, List
from warnings import warn

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch

import numpy as np

from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.process.utils.dt_tracker import DTTracker
from torch_kalman.utils import split_flat
from torch_kalman.process.utils.fourier import fourier_tensor


class FourierSeason(Process):
    def __init__(self,
                 id: str,
                 seasonal_period: Union[int, float],
                 K: int,
                 decay: Union[bool, Tuple[float, float]] = False,
                 season_start: Union[str, None, bool] = None,
                 dt_unit: Optional[str] = None):

        # handle datetimes:
        self.dt_tracker = DTTracker(season_start=season_start, dt_unit=dt_unit)

        # season structure:
        self.seasonal_period = seasonal_period
        self.K = K

        if decay:
            assert not isinstance(decay, bool), "decay should be floats of bounds (or False for no decay)"
            assert decay[0] > 0. and decay[1] <= 1.0
            self.decay = Bounded(*decay)
        else:
            self.decay = None

        #
        state_elements, transitions = self._setup(decay=decay)

        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

        if self.dt_tracker.start_datetime:
            self.expected_batch_kwargs.append('start_datetimes')

    def _setup(self, decay: bool) -> Tuple[List[str], Dict[str, Dict]]:
        state_elements = []
        transitions = {}
        for r in range(self.K):
            for c in range(2):
                element_name = f"{r},{c}"
                state_elements.append(element_name)
                if decay:
                    transitions[element_name] = {element_name: lambda pfb: pfb.process.decay.value}
                else:
                    transitions[element_name] = {element_name: 1.0}

        return state_elements, transitions

    def parameters(self) -> Generator[Parameter, None, None]:
        if self.decay is not None:
            yield self.decay.parameter

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        raise NotImplementedError

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  start_datetimes: Optional[np.ndarray] = None) -> ProcessForBatch:
        # super:
        for_batch = super().for_batch(num_groups, num_timesteps)

        self._modify_batch(for_batch, start_datetimes)

        return for_batch

    def _modify_batch(self, proc_for_batch: ProcessForBatch, start_datetimes: Optional[np.ndarray]) -> None:
        raise NotImplementedError


class FourierSeasonDynamic(FourierSeason):
    def _setup(self, decay: bool) -> Tuple[List[str], Dict[str, Dict]]:
        state_elements, transitions = super()._setup(decay=decay)

        # all fourier components transition into position:
        transitions['position'] = {se: None for se in state_elements}

        # add position. note that it doesn't transition into itself
        state_elements.append('position')
        return state_elements, transitions

    def _modify_batch(self, proc_for_batch: ProcessForBatch, start_datetimes: Optional[np.ndarray]) -> None:
        # determine the delta (integer time accounting for different groups having different start datetimes)
        delta = self.dt_tracker.get_delta(proc_for_batch.num_groups, proc_for_batch.num_timesteps,
                                          start_datetimes=start_datetimes)

        # determine season:
        season = delta % self.seasonal_period

        # generate the fourier tensor:
        fourier_tens = fourier_tensor(time=Tensor(season), seasonal_period=self.seasonal_period, K=self.K)

        for state_element in self.state_elements:
            if state_element == 'position':
                continue
            r, c = (int(x) for x in state_element.split(sep=","))
            values = split_flat(fourier_tens[:, :, r, c], dim=1, clone=True)
            proc_for_batch.set_transition(from_element=state_element, to_element='position', values=values)

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements[:-1]

    def add_measure(self, measure: str, state_element='position', value=1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)


class FourierSeasonFixed(FourierSeason):
    def _modify_batch(self, proc_for_batch: ProcessForBatch, start_datetimes: Optional[np.ndarray]) -> None:
        # determine the delta (integer time accounting for different groups having different start datetimes)
        delta = self.dt_tracker.get_delta(proc_for_batch.num_groups, proc_for_batch.num_timesteps,
                                          start_datetimes=start_datetimes)

        # determine season:
        season = delta % self.seasonal_period

        # generate the fourier tensor:
        fourier_tens = fourier_tensor(time=Tensor(season), seasonal_period=self.seasonal_period, K=self.K)

        for measure in self.measures:
            for state_element in self.state_elements:
                r, c = (int(x) for x in state_element.split(sep=","))
                values = split_flat(fourier_tens[:, :, r, c], dim=1, clone=True)
                proc_for_batch.add_measure(measure=measure, state_element=state_element, values=values)

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return []

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        for state_element in self.state_elements:
            super().add_measure(measure=measure, state_element=state_element, value=None)
