from typing import Generator, Tuple, Union, Callable, Sequence

from torch.nn import Parameter

from torch_kalman.process import Process
from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.utils import itervalues_sorted_keys


class LocalTrend(Process):
    def __init__(self,
                 id: str,
                 decay_velocity: Union[bool, Tuple[float, float]] = (.95, 1.00),
                 decay_position: Union[bool, Tuple[float, float]] = False):
        # state-elements:
        state_elements = ['position', 'velocity']

        # transitions:
        transitions = {'position': {'position': None, 'velocity': 1.0},
                       'velocity': {'velocity': None}}

        self.decayed_transitions = {}
        if decay_position:
            assert not isinstance(decay_position, bool), "decay_position should be floats of bounds (or False for no decay)"
            self.decayed_transitions['position'] = Bounded(*decay_position)
            assert decay_position[0] > 0. and decay_position[1] <= 1.
            transitions['position']['position'] = lambda pfb: pfb.process.decayed_transitions['position'].value
        else:
            transitions['position']['position'] = 1.0

        if decay_velocity:
            assert not isinstance(decay_velocity, bool), "decay_velocity should be floats of bounds (or False for no decay)"
            self.decayed_transitions['velocity'] = Bounded(*decay_velocity)
            assert decay_velocity[0] > 0. and decay_velocity[1] <= 1.
            transitions['velocity']['velocity'] = lambda pfb: pfb.process.decayed_transitions['velocity'].value
        else:
            transitions['velocity']['velocity'] = 1.0

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    def parameters(self) -> Generator[Parameter, None, None]:
        for transition in itervalues_sorted_keys(self.decayed_transitions):
            yield transition.parameter

    def add_measure(self, measure: str, state_element: str = 'position', value: Union[float, Callable, None] = 1.0) -> None:
        # default values (position, 1.0)
        super().add_measure(measure=measure, state_element=state_element, value=value)

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements
