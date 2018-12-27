from typing import Optional, Union, Generator, Dict, Sequence, Tuple, Callable

import numpy as np
import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process.utils.fourier import fourier_tensor
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.processes.season.base import DateAware
from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.utils import split_flat


class Season(DateAware):
    def __init__(self,
                 id: str,
                 seasonal_period: int,
                 season_duration: int = 1,
                 decay: Union[bool, Tuple[float, float]] = False,
                 *args, **kwargs):
        """
        Process representing discrete seasons.

        :param id: Unique name for this process
        :param seasonal_period: The number of seasons (e.g. 7 for day_in_week).
        :param season_duration: The length of each season, default 1 time-step.
        :param decay: Analogous to dampening a trend -- the state will revert to zero as we get further from the last
        observation. This can be useful if two processes are capturing the same seasonal pattern: one can be more flexible,
        but with decay have a tendency to revert to zero, while the other is less variable but extrapolates into the future.
        """

        self.seasonal_period = seasonal_period
        self.season_duration = season_duration

        # state-elements:
        self.measured_name = 'measured'
        pad_n = len(str(seasonal_period))
        state_elements = [self.measured_name] + [str(i).rjust(pad_n, "0") for i in range(1, seasonal_period)]

        # transitions are placeholder, filled in w/batch
        transitions = {}
        for i in range(len(state_elements)):
            current = state_elements[i]
            transitions[current] = {current: None}
            if i > 0:
                prev = state_elements[i - 1]
                transitions[current][prev] = None
                transitions[self.measured_name][prev] = None

        if decay:
            assert not isinstance(decay, bool), "decay should be floats of bounds (or False for no decay)"
            assert decay[0] > 0. and decay[1] <= 1.0
            self.decay = Bounded(*decay)
        else:
            self.decay = None

        # super:
        super().__init__(id=id,
                         state_elements=state_elements,
                         transitions=transitions,
                         *args, **kwargs)

        # writing transition-matrix is slow, no need to do it repeatedly:
        self.transition_cache = {}

    def add_measure(self,
                    measure: str,
                    state_element: str = 'measured',
                    value: Union[float, Callable, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def parameters(self) -> Generator[Parameter, None, None]:
        if self.decay is not None:
            yield self.decay.parameter

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  start_datetimes: Optional[np.ndarray] = None) -> ProcessForBatch:

        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps, start_datetimes=start_datetimes)
        delta = self.get_delta(for_batch.num_groups, for_batch.num_timesteps, start_datetimes=start_datetimes)

        in_transition = (delta % self.season_duration) == (self.season_duration - 1)

        transitions = dict()
        transitions['to_next_state'] = torch.from_numpy(in_transition.astype('float32'))
        transitions['to_self'] = 1 - transitions['to_next_state']
        transitions['to_measured'] = -transitions['to_next_state']
        transitions['from_measured_to_measured'] = torch.from_numpy(np.where(in_transition, -1., 1.).astype('float32'))
        for k in transitions.keys():
            transitions[k] = split_flat(transitions[k], dim=1)
            if self.decay is not None:
                transitions[k] = [x * self.decay.value for x in transitions[k]]

        # this is convoluted, but the idea is to manipulate the transitions so that we use one less degree of freedom than
        # the number of seasons, by having the 'measured' state be equal to -sum(all others)
        for i in range(1, len(self.state_elements)):
            current = self.state_elements[i]
            prev = self.state_elements[i - 1]

            if prev == self.measured_name:  # measured requires special-case
                to_measured = transitions['from_measured_to_measured']
            else:
                to_measured = transitions['to_measured']

            for_batch.set_transition(from_element=prev, to_element=current, values=transitions['to_next_state'])
            for_batch.set_transition(from_element=prev, to_element=self.measured_name, values=to_measured)

            # from state to itself:
            for_batch.set_transition(from_element=current, to_element=current, values=transitions['to_self'])

        return for_batch

    def initial_state_for_batch(self, num_groups: int, start_datetimes: Optional[np.ndarray] = None):
        delta = self.get_delta(num_groups, 1, start_datetimes=start_datetimes).squeeze(1)

        init_mean, init_cov = self.initial_state()
        season_shift = (np.floor(delta / self.season_duration) % self.seasonal_period).astype('int')

        means = [torch.cat([init_mean[-shift:], init_mean[:-shift]]) for shift in season_shift]
        means = torch.stack(means)

        covs = init_cov.expand(num_groups, -1, -1)

        return means, covs

    def initial_state_means_for_batch(self,
                                      parameters: Parameter,
                                      batch_size: int,
                                      start_datetimes: Optional[np.ndarray] = None) -> Tensor:

        delta = self.get_delta(batch_size, 1, start_datetimes=start_datetimes).squeeze(1)
        season_shift = (np.floor(delta / self.season_duration) % self.seasonal_period).astype('int')
        means = [torch.cat([parameters[-shift:], parameters[:-shift]]) for shift in season_shift]
        return torch.stack(means, 0)
