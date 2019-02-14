from typing import Union, Tuple, Generator, Sequence, Optional

import torch

from torch_kalman.process import Process

from torch_kalman.process.utils.bounded import Bounded


class LocalLevel(Process):
    def __init__(self,
                 id: str,
                 decay: Union[bool, Tuple[float, float]] = False):
        super().__init__(id=id, state_elements=['position'])

        self.decay: Optional[Bounded] = None
        if decay:
            assert not isinstance(decay, bool), "decay should be floats of bounds (or False for no decay)"
            assert decay[0] >= -1. and decay[1] <= 1.
            self.decay = Bounded(*decay)
            self._set_transition(from_element='position',
                                 to_element='position',
                                 value=self.decay.get_value,
                                 inv_link=False)
        else:
            self._set_transition(from_element='position', to_element='position', value=1.)

    def add_measure(self, measure: str):
        self._set_measure(measure=measure, state_element='position', value=1.0)

    def parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        if self.decay is not None:
            yield self.decay.parameter

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self.state_elements
