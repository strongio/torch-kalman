import torch
from torch import Tensor
from torch.nn import Parameter

from typing import Optional, Tuple

from torch_kalman.process.processes.season.discrete import Season
import numpy as np


class NestedSeason(Season):
    """
    A season that's nested in another season. For example, an 'hour-in-day' season might have a structure whose 'amplitude'
    changes based on the level of an outer 'day-in-week' seasonality. In effect, this allows for a multiplicative effect for
    these seasons, without (or in addition to) any multiplicative effect in the rest of the model/error-terms.
    """

    def __init__(self, outer_season_id: int, *args, **kwargs):
        self.outer_season_id = outer_season_id
        self.outer_season_state_idx = None
        self.outer_state_to_measure_param = Parameter(torch.zeros(1))
        super().__init__(*args, **kwargs)
        self.expected_batch_kwargs.append('state_prediction')

    def initial_state(self,
                      batch_size: int,
                      start_datetimes: Optional[np.ndarray] = None,
                      time: Optional[int] = None,
                      state_prediction: Optional['StateBelief'] = None) -> Tuple[Tensor, Tensor]:
        return super().initial_state(batch_size, start_datetimes, time)

    def link_to_design(self, design: 'Design'):
        if self.outer_season_state_idx is not None:
            raise RuntimeError(f"Tried to call `link_to_design` on process '{self.id}', but this process was already linked "
                               f"to a design.")

        for i, (process_name, state_element) in enumerate(design.all_state_elements()):
            if process_name == self.outer_season_id and state_element == 'measured':
                self.outer_season_state_idx = i
                break
        if self.outer_season_state_idx is None:
            raise RuntimeError(f"'{self.id}' expected a process named '{self.outer_season_id}' (with a measure named "
                               f"'state_element'), but didn't find it in this design.")

    def parameters(self):
        for param in super().parameters():
            yield param
        yield self.outer_state_to_measure_param

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str):
        super().add_measure(measure=measure, state_element='measured', value=None)

    def for_batch(self,
                  batch_size: int,
                  time: Optional[int] = None,
                  start_datetimes: Optional[np.ndarray] = None,
                  state_prediction: Optional['StateBelief'] = None,
                  cache: bool = True) -> 'ProcessForBatch':

        # do the usual season stuff:
        for_batch = super().for_batch(batch_size=batch_size, time=time, start_datetimes=start_datetimes, cache=cache)

        # link outer-season state to inner-state measurement-multi:
        measure_values = 1.0 + state_prediction.means[:, self.outer_season_state_idx] * self.outer_state_to_measure_param

        for measure in self.measures():
            for_batch.add_measure(measure=measure, state_element='measured', values=measure_values)

        return for_batch
