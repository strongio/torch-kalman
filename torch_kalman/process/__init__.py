from typing import Generator, Sequence, Dict, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.state_belief import StateBelief


class Process:
    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 transitions: Dict[str, Dict[str, Union[float, None]]]):
        """

        :param id: A unique string identifying the process.
        :param state_elements: A sequence of names for the states that make up this process.
        :param transitions: A dictionary whose structure is easier to understand by example. For example:
            `transitions=dict(position = dict(position = 1.0, velocity = 1.0))`
        is designed to be read like an equation:
            position_t2 = 1.0 * position_t1 + 1.0 * velocity_t1
        That is, we specify a Dict whose keys are states being transitioned *to*, and whose values are themselves Dicts
        specifying the states *from* which we are transitioning. When specifying None for a value, this means that the value
        needs to be generated on a per-batch basis, via ProcessForBatch.set_transition.
        """

        self.id = str(id)
        self._state_element_idx = None

        # measures:
        self.measures = {}

        # state elements:
        assert len(state_elements) == len(set(state_elements)), "Duplicate `state_elements` now allowed."
        self.state_elements = state_elements

        # transitions:
        self.transitions = transitions
        for to_el, from_els in self.transitions.items():
            assert to_el in self.state_elements, f"`{to_el}` is in transitions but not state_elements"
            for from_el in from_els.keys():
                assert to_el in self.state_elements, f"`{from_el}` is in transitions but not state_elements"

        # expected kwargs
        self.expected_batch_kwargs = ()

        # if no per-batch modification, can avoid repeated computations:
        self.F_base = None

    def align_initial_state(self, state_belief: StateBelief, **kwargs) -> StateBelief:
        return state_belief

    def add_measure(self, measure: str, state_element: str, value: Union[float, Tensor, None]) -> None:
        assert state_element in self.state_elements

        key = (measure, state_element)

        if key in self.measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process.")
        self.measures[key] = value

    def parameters(self) -> Generator[Parameter, None, None]:
        raise NotImplementedError

    def covariance(self) -> Covariance:
        raise NotImplementedError

    @property
    def state_element_idx(self) -> dict:
        if self._state_element_idx is None:
            self._state_element_idx = {el: i for i, el in enumerate(self.state_elements)}
        return self._state_element_idx

    def for_batch(self, batch_size: int, **kwargs) -> 'ProcessForBatch':
        return ProcessForBatch(process=self, batch_size=batch_size)


class ProcessForBatch:
    def __init__(self, process: Process, batch_size: int):
        self.batch_size = batch_size
        self.process = process

        # a bit over-protective: batch process gets these, but they're copies so no one will accidentally modify originals
        self.id = str(self.process.id)
        self.state_elements = list(self.process.state_elements)
        self.state_element_idx = dict(self.process.state_element_idx)

        # transitions that are specific to this batch, not the process generally:
        self.batch_transitions = {}
        self.batch_measures = {}

    def F(self) -> Tensor:
        if not self.batch_transitions and self.process.F_base is not None:
            return self.process.F_base.expand(self.batch_size, -1, -1)

        # fill in template:
        F = torch.zeros(size=(self.batch_size, len(self.process.state_elements), len(self.process.state_elements)))
        for to_el, from_els in self.transitions().items():
            for from_el, values in from_els.items():
                r, c = self.process.state_element_idx[to_el], self.process.state_element_idx[from_el]
                if values is None:
                    raise ValueError(f"The transition from '{from_el}' to '{to_el}' is None, which means this process "
                                     f"('{self.process.__class__.__name__}') requires you set it on a per-batch basis using "
                                     f"the `set_transition` method.")
                F[:, r, c] = values

        if not self.batch_transitions:
            self.process.F_base = F[0]
        return F

    def Q(self) -> Tensor:
        # generate covariance:
        cov = self.process.covariance()
        # expand for batch:
        return cov.expand(self.batch_size, -1, -1)

    def set_transition(self, from_element: str, to_element: str, values: Tensor) -> None:
        assert len(values) == 1 or len(values) == self.batch_size
        assert from_element in self.process.state_elements
        assert to_element in self.process.state_elements

        if to_element in self.process.transitions.keys():
            if self.process.transitions[to_element].get(from_element, None):
                raise ValueError(f"The transition from '{from_element}' to '{to_element}' was already set for this Process,"
                                 f" so can't give it batch-specific values.")

        if to_element not in self.batch_transitions.keys():
            self.batch_transitions[to_element] = {}
        elif from_element in self.batch_transitions[to_element]:
            raise ValueError(f"The transition from '{from_element}' to '{to_element}' was already set for this batch,"
                             f" so can't set it again.")

        self.batch_transitions[to_element][from_element] = values

    def add_measure(self, measure, state_element, values=None):
        if values is not None:
            assert len(values) == 1 or len(values) == self.batch_size
        assert state_element in self.process.state_elements

        key = (measure, state_element)

        if self.process.measures.get(key, None):
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process, cannot modify.")

        if key in self.batch_measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this batch-process.")

        self.batch_measures[key] = values

    def transitions(self):
        # need to be careful to update "deeply", and also not modify originals

        out = {}
        # non-batch:
        for to_el, from_els in self.process.transitions.items():
            out[to_el] = dict()
            for from_el, value in from_els.items():
                out[to_el][from_el] = value

        # batch:
        for to_el, from_els in self.batch_transitions.items():
            if to_el not in out.keys():
                out[to_el] = dict()
            for from_el, values in from_els.items():
                out[to_el][from_el] = values

        return out

    def measures(self):
        # don't need to be as careful as w/self.transitions, since dicts of values, not dicts of dicts
        return {**self.process.measures, **self.batch_measures}
