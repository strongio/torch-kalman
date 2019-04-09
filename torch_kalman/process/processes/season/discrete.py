from typing import Optional, Union, Tuple, Sequence

import numpy as np
import torch

from torch import Tensor
from torch.nn import Parameter, ParameterDict

from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.process.utils.dt_tracker import DTTracker
from torch_kalman.utils import split_flat


class Season(Process):

    def __init__(self,
                 id: str,
                 seasonal_period: int,
                 season_duration: int = 1,
                 decay: Union[bool, Tuple[float, float]] = False,
                 season_start: Optional[str] = None,
                 dt_unit: Optional[str] = None):
        """
        Process representing discrete seasons.

        :param id: Unique name for this process
        :param seasonal_period: The number of seasons (e.g. 7 for day_in_week).
        :param season_duration: The length of each season, default 1 time-step.
        :param decay: Analogous to dampening a trend -- the state will revert to zero as we get further from the last
        observation. This can be useful if two processes are capturing the same seasonal pattern: one can be more flexible,
        but with decay have a tendency to revert to zero, while the other is less variable but extrapolates into the future.
        :param season_start: A string that can be parsed into a datetime by `numpy.datetime64`. See DTTracker.
        :param dt_unit: Currently supports {'Y', 'D', 'h', 'm', 's'}. 'W' is experimentally supported.
        """

        # handle datetimes:
        self.dt_tracker = DTTracker(season_start=season_start, dt_unit=dt_unit, process_id=id)

        #
        self.seasonal_period = seasonal_period
        self.season_duration = season_duration

        # state-elements:
        self.measured_name = 'measured'
        pad_n = len(str(seasonal_period))
        super().__init__(id=id,
                         state_elements=[self.measured_name] + [str(i).rjust(pad_n, "0") for i in range(1, seasonal_period)])

        # transitions are placeholders, filled in w/batch
        for i, current in enumerate(self.state_elements):
            self._set_transition(from_element=current, to_element=current, value=0.)
            if i > 0:
                prev = self.state_elements[i - 1]
                self._set_transition(from_element=prev, to_element=current, value=0.)
                if i > 1:
                    self._set_transition(from_element=prev, to_element=self.measured_name, value=0.)

        if decay:
            assert not isinstance(decay, bool), "decay should be floats of bounds (or False for no decay)"
            assert decay[0] > 0. and decay[1] <= 1.0
            self.decay = Bounded(*decay)
        else:
            self.decay = None

    def add_measure(self, measure: str):
        self._set_measure(measure=measure, state_element='measured', value=1.0)

    def param_dict(self) -> ParameterDict:
        p = ParameterDict()
        if self.decay is not None:
            p['decay'] = self.decay.parameter
        return p

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return [self.measured_name]

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  start_datetimes: Optional[np.ndarray] = None) -> ProcessForBatch:

        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps)

        delta = self.dt_tracker.get_delta(for_batch.num_groups, for_batch.num_timesteps, start_datetimes=start_datetimes)

        in_transition = (delta % self.season_duration) == (self.season_duration - 1)

        transitions = dict()
        transitions['to_next_state'] = torch.from_numpy(in_transition.astype('float32'))
        transitions['to_self'] = 1 - transitions['to_next_state']
        transitions['to_measured'] = -transitions['to_next_state']
        transitions['from_measured_to_measured'] = torch.from_numpy(np.where(in_transition, -1., 1.).astype('float32'))
        for k in transitions.keys():
            transitions[k] = split_flat(transitions[k], dim=1, clone=True)
            if self.decay is not None:
                decay_value = self.decay.get_value()
                transitions[k] = [x * decay_value for x in transitions[k]]

        # this is convoluted, but the idea is to manipulate the transitions so that we use one less degree of freedom than
        # the number of seasons, by having the 'measured' state be equal to -sum(all others)
        for i in range(1, len(self.state_elements)):
            current = self.state_elements[i]
            prev = self.state_elements[i - 1]

            if prev == self.measured_name:  # measured requires special-case
                to_measured = transitions['from_measured_to_measured']
            else:
                to_measured = transitions['to_measured']

            for_batch.adjust_transition(from_element=prev, to_element=current, adjustment=transitions['to_next_state'])
            for_batch.adjust_transition(from_element=prev, to_element=self.measured_name, adjustment=to_measured)

            # from state to itself:
            for_batch.adjust_transition(from_element=current, to_element=current, adjustment=transitions['to_self'])

        return for_batch

    def initial_state_means_for_batch(self,
                                      parameters: Parameter,
                                      num_groups: int,
                                      start_datetimes: Optional[np.ndarray] = None) -> Tensor:

        delta = self.dt_tracker.get_delta(num_groups, 1, start_datetimes=start_datetimes).squeeze(1)
        season_shift = (np.floor(delta / self.season_duration) % self.seasonal_period).astype('int')
        means = [torch.cat([parameters[-shift:], parameters[:-shift]]) for shift in season_shift]
        return torch.stack(means, 0)
