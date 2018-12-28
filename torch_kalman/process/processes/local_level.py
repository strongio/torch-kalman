from typing import Union, Tuple, Generator, Callable, Sequence

import torch

from torch_kalman.process import Process
from torch_kalman.process.utils.bounded import Bounded


class LocalLevel(Process):
    def __init__(self,
                 id: str,
                 decay: Union[bool, Tuple[float, float]] = False):
        state_elements = ['position']

        if decay:
            assert not isinstance(decay, bool), "decay should be floats of bounds (or False for no decay)"
            assert decay[0] >= -1. and decay[1] <= 1.
            self.decay = Bounded(*decay)
            transitions = {'position': {'position': lambda proc_for_batch: proc_for_batch.process.decay.value}}
        else:
            self.decay = None
            transitions = {'position': {'position': 1.0}}

        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    def add_measure(self, measure: str, state_element: str = 'position', value: Union[float, Callable, None] = 1.0) -> None:
        # default values (position, 1.0)
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        if self.decay is not None:
            yield self.decay.parameter

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements
