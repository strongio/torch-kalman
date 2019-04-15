from typing import Generator, Tuple, Union, Callable, Sequence

from torch.nn import Parameter, ParameterDict

from torch_kalman.process import Process
from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.utils import itervalues_sorted_keys


class LocalTrend(Process):
    def __init__(self,
                 id: str,
                 decay_velocity: Union[bool, Tuple[float, float]] = (.95, 1.00),
                 decay_position: Union[bool, Tuple[float, float]] = False,
                 multi: float = 1.0):

        super().__init__(id=id, state_elements=['position', 'velocity'])
        self._set_transition(from_element='velocity', to_element='position', value=multi)

        self.decayed_transitions = {}
        if decay_position:
            assert not isinstance(decay_position, bool), "decay_position should be floats of bounds (or False for no decay)"
            assert decay_position[0] > 0. and decay_position[1] <= 1.
            self.decayed_transitions['position'] = Bounded(*decay_position)
            self._set_transition(from_element='position',
                                 to_element='position',
                                 value=self.decayed_transitions['position'].get_value,
                                 inv_link=False)
        else:
            self._set_transition(from_element='position', to_element='position', value=1.0)

        if decay_velocity:
            assert not isinstance(decay_velocity, bool), "decay_velocity should be floats of bounds (or False for no decay)"
            assert decay_velocity[0] > 0. and decay_velocity[1] <= 1.
            self.decayed_transitions['velocity'] = Bounded(*decay_velocity)
            self._set_transition(from_element='velocity',
                                 to_element='velocity',
                                 value=self.decayed_transitions['velocity'].get_value,
                                 inv_link=False)
        else:
            self._set_transition(from_element='velocity', to_element='velocity', value=1.0)

    def param_dict(self) -> ParameterDict:
        p = ParameterDict()
        p.update({name: transition.parameter for name, transition in self.decayed_transitions.items()})
        return p

    def add_measure(self, measure: str) -> 'LocalTrend':
        self._set_measure(measure=measure, state_element='position', value=1.0)
        return self

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements
