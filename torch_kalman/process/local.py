from typing import Tuple, List, Optional, Dict

import torch

from torch import nn, Tensor

from .base import Process
from .utils import SimpleTransition, ReturnValues


class LocalLevel(Process):
    def get_groupwise_kwargs(self, *args, **kwargs) -> Dict[str, Tensor]:
        return {}

    def get_timewise_kwargs(self, *args, **kwargs) -> Dict[str, Tensor]:
        return {}

    def __init__(self, id: str, decay: Optional[Tuple[float, float]] = None):
        """
        :param id: A unique identifier for this process.
        :param decay: If the process has decay, then the random walk will tend towards zero as we forecast out further
        (note that this means you should center your time-series, or you should include another process that does not
        have this property). Decay can be between 0 and 1, but values < .50 (or even .90) can often be too rapid and
        you will run into trouble with vanishing gradients. When passing a pair of floats, the nn.Module will assign a
        parameter representing the decay as a learned parameter somewhere between these bounds.
        """
        se = 'position'
        if decay:
            transitions = nn.ModuleDict()
            transitions[f'{se}->{se}'] = SimpleTransition(decay or (None, 1.0))
        else:
            transitions = {se: torch.ones(1)}
        super(LocalLevel, self).__init__(
            id=id,
            state_elements=[se],
            f_modules=transitions if decay else None,
            f_tensors=None if decay else transitions,
            h_tensor=torch.tensor([1.])
        )


class LocalTrend(Process):

    def get_groupwise_kwargs(self, *args, **kwargs) -> Dict[str, Tensor]:
        return {}

    def get_timewise_kwargs(self, *args, **kwargs) -> Dict[str, Tensor]:
        return {}

    def __init__(self,
                 id: str,
                 decay_velocity: Optional[Tuple[float, float]] = (.95, 1.00),
                 decay_position: Optional[Tuple[float, float]] = None,
                 velocity_multi: float = 1.0):
        """
        :param id: A unique identifier for this process.
        :param decay_velocity: If set, then the trend will decay to zero as we forecast out further. The default is
        to allow the trend to decay somewhere between .95 (moderate decay) and 1.00 (no decay), with the exact value
         being a learned parameter.
        :param decay_position: See `decay` in `LocalLevel`. Default is no decay.
        :param velocity_multi: A multiplier on the velocity, so that
        `next_position = position + velocity_multi * velocity`. A value of << 1.0 can be helpful since the
        trend has such a large effect on the prediction, so that large values can lead to exploding predictions.
        """

        # define transitions:
        f_modules = nn.ModuleDict()
        f_tensors = {}

        if decay_position is None:
            f_tensors['position->position'] = torch.ones(1)
        else:
            f_modules['position->position'] = SimpleTransition(decay_position)
        if decay_velocity is None:
            f_tensors['velocity->velocity'] = torch.ones(1)
        else:
            f_modules['velocity->velocity'] = SimpleTransition(decay_velocity)

        assert velocity_multi <= 1.
        f_tensors['velocity->position'] = torch.ones(1) * velocity_multi

        super(LocalTrend, self).__init__(
            id=id,
            state_elements=['position', 'velocity'],
            f_modules=f_modules,
            f_tensors=f_tensors,
            h_tensor=torch.tensor([1., 0.])
        )
