from typing import TypeVar

import torch

from torch_kalman.design import Design
from torch_kalman.design.for_batch import DesignForBatch
from torch_kalman.state_belief import Gaussian, StateBelief


class Simulation:
    def __init__(self, design: Design):
        assert not design.requires_grad
        self.design = design

    @property
    def family(self) -> TypeVar('StateBelief'):
        return Gaussian

    def simulate(self, num_groups: int, num_timesteps: int, **kwargs):
        design_for_batch = self.design.for_batch(num_groups=num_groups, num_timesteps=num_timesteps, **kwargs)
        initial_state = self.family(*design_for_batch.get_block_diag_initial_state(num_groups=num_groups))
        return initial_state.simulate(design_for_batch=design_for_batch, **kwargs)
