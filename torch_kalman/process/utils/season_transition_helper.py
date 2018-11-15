from typing import Sequence, Dict, Union

from numpy import ndarray, where
from torch import Tensor

from torch_kalman.process.for_batch import ProcessForBatch


class SeasonTransitionHelper:
    def __init__(self,
                 state_elements: Sequence[str]):
        self.state_elements = state_elements
        self.measured_name = self.state_elements[0]

    def initialize(self) -> Dict[str, Dict[str, Union[float, None]]]:
        seasonal_period = len(self.state_elements)

        # transitions are placeholder, filled in w/batch
        transitions = dict()
        for i in range(seasonal_period):
            current = self.state_elements[i]
            transitions[current] = {current: None}
            if i > 0:
                prev = self.state_elements[i - 1]
                transitions[current][prev] = None
                transitions[self.measured_name][prev] = None

        return transitions

    def for_batch(self,
                  for_batch: ProcessForBatch,
                  in_transition: ndarray,
                  ) -> Dict[str, Dict[str, Tensor]]:
        to_next_state = Tensor(in_transition.astype('float'))
        to_self = 1 - to_next_state

        assert len(for_batch.batch_transitions) == 0, "Expected an empty ProcessForBatch"

        for i in range(1, len(self.state_elements)):
            current = self.state_elements[i]
            prev = self.state_elements[i - 1]

            # from state to next state
            for_batch.set_transition(from_element=prev, to_element=current, values=to_next_state)

            # from state to measured:
            if prev == self.measured_name:  # first requires special-case
                to_measured = Tensor(where(in_transition, -1.0, 1.0))
            else:
                to_measured = -to_next_state
            for_batch.set_transition(from_element=prev, to_element=self.measured_name, values=to_measured)

            # from state to itself:
            for_batch.set_transition(from_element=current, to_element=current, values=to_self)

        # shouldn't ever come up, but best practice not to modify input in-place:
        out = dict(for_batch.batch_transitions)
        for_batch.batch_transitions = {}

        return out
