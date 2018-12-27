from typing import Generator, Tuple, Optional, Union, Dict, Sequence

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance

from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.processes.season.base import DateAware

import numpy as np

from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.utils import itervalues_sorted_keys, split_flat
from torch_kalman.process.utils.fourier import fourier_tensor


class FourierSeason(DateAware):
    def __init__(self,
                 id: str,
                 seasonal_period: Union[int, float],
                 K: int,
                 process_variance: bool = False,
                 decay: Union[bool, Tuple[float, float]] = False,
                 **kwargs):
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

        self._dynamic_state_elements = state_elements if process_variance else []

        super().__init__(id=id, state_elements=state_elements, transitions=transitions, **kwargs)

    def parameters(self) -> Generator[Parameter, None, None]:
        if self.decay is not None:
            yield self.decay.parameter

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self._dynamic_state_elements

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        for state_element in self.state_elements:
            super().add_measure(measure=measure, state_element=state_element, value=None)

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  start_datetimes: Optional[np.ndarray] = None) -> ProcessForBatch:
        # super:
        for_batch = super().for_batch(num_groups, num_timesteps)

        # determine the delta (integer time accounting for different groups having different start datetimes)
        delta = self.get_delta(for_batch.num_groups, for_batch.num_timesteps, start_datetimes=start_datetimes)

        # determine season:
        season = delta % self.seasonal_period

        # generate the fourier tensor:
        fourier_tens = fourier_tensor(time=Tensor(season), seasonal_period=self.seasonal_period, K=self.K)

        for measure in self.measures:
            for state_element in self.state_elements:
                r, c = (int(x) for x in state_element.split(sep=","))
                values = split_flat(fourier_tens[:, :, r, c], dim=1)
                for_batch.add_measure(measure=measure, state_element=state_element, values=values)

        return for_batch
