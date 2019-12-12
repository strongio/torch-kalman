from typing import Tuple, Union

from torch.nn import ParameterDict

from torch_kalman.process import Process
from torch_kalman.process.utils.bounded import Bounded


class LocalTrend(Process):
    def __init__(self,
                 id: str,
                 decay_velocity: Union[bool, Tuple[float, float]] = (.95, 1.00),
                 decay_position: Union[bool, Tuple[float, float]] = False,
                 multi: float = 1.0):

        super().__init__(id=id, state_elements=['position', 'velocity'])

        self.decayed_transitions = {}

        # does position regress towards zero?
        if decay_position:
            assert decay_position[0] > 0. and decay_position[1] <= 1.
            self.decayed_transitions['position'] = Bounded(*decay_position)
            self._set_transition(
                from_element='position',
                to_element='position',
                value=self.decayed_transitions['position'].get_value
            )
        else:
            self._set_transition(from_element='position', to_element='position', value=1.0)

        # does velocity regress towards zero?
        if decay_velocity:
            assert decay_velocity[0] > 0. and decay_velocity[1] <= 1.
            self.decayed_transitions['velocity'] = Bounded(*decay_velocity)
            self._set_transition(
                from_element='velocity',
                to_element='velocity',
                value=self.decayed_transitions['velocity'].get_value
            )
        else:
            self._set_transition(from_element='velocity', to_element='velocity', value=1.0)

        # setting a low arbitrary multiplier on velocity's impact can sometimes be helpful for training in practice:
        self._set_transition(from_element='velocity', to_element='position', value=multi)

    def param_dict(self) -> ParameterDict:
        p = ParameterDict()
        for k in ('position', 'velocity'):
            if k in self.decayed_transitions:
                p[k] = self.decayed_transitions[k].parameter
        return p

    def add_measure(self, measure: str) -> 'LocalTrend':
        self._set_measure(measure=measure, state_element='position', value=1.0)
        return self
