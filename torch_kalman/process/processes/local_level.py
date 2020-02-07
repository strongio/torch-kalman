from typing import Union, Tuple, Sequence, Optional

import torch

from torch_kalman.process import Process

from torch_kalman.process.utils.bounded import Bounded


class LocalLevel(Process):
    """
    A simple random walk.
    """
    def __init__(self,
                 id: str,
                 decay: Union[bool, Tuple[float, float]] = False):
        """
        :param id: A unique identifier for this process.
        :param decay: If the process has decay, then the random walk will tend towards zero as we forecast out further
        (note that this means you should center your time-series, or you should include another process that does not
        have this property). Decay can be between 0 and 1, but values < .50 (or even .90) can often be too rapid and
        you will run into trouble with vanishing gradients. When passing a pair of floats, the nn.Module will assign a
        parameter representing the decay as a learned parameter somewhere between these bounds.
        """
        super().__init__(id=id, state_elements=['position'])

        self.decay = None
        if decay:
            assert decay[0] >= -1. and decay[1] <= 1.
            self.decay = Bounded(*decay)
            self._set_transition(
                from_element='position',
                to_element='position',
                value=self.decay.get_value
            )
        else:
            self._set_transition(from_element='position', to_element='position', value=1.)

    def add_measure(self, measure: str) -> 'LocalLevel':
        self._set_measure(measure=measure, state_element='position', value=1.0)
        return self

    def param_dict(self) -> torch.nn.ParameterDict:
        p = torch.nn.ParameterDict()
        if self.decay is not None:
            p['decay'] = self.decay.parameter
        return p
