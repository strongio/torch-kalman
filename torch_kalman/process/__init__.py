from typing import Generator, Sequence, Dict

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance


class Process:
    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 transitions: Dict[str, Dict[str, float]]):

        self.id = str(id)
        self._state_element_idx = None

        # state elements:
        self.state_elements = state_elements

        # transitions:
        self.transitions = transitions
        for to_el, from_els in self.transitions.items():
            assert to_el in self.state_elements, f"{to_el} is in transitions but not state_elements"
            for from_el in from_els.keys():
                assert to_el in self.state_elements, f"{from_el} is in transitions but not state_elements"

    def parameters(self) -> Generator[Parameter, None, None]:
        raise NotImplementedError

    def covariance(self) -> Covariance:
        raise NotImplementedError

    @property
    def state_element_idx(self) -> dict:
        if self._state_element_idx is None:
            self._state_element_idx = {el: i for i, el in enumerate(self.state_elements)}
        return self._state_element_idx

    def for_batch(self, batch_size):
        return ProcessForBatch(process=self, batch_size=batch_size)

    @property
    def measurable_state(self) -> str:
        raise NotImplementedError


class ProcessForBatch:
    def __init__(self, process: Process, batch_size: int):
        self.batch_size = batch_size
        self.process = process

        # a bit over-protective: batch process gets these, but they're copies so no one will accidentally modify originals
        self.state_elements = list(self.process.state_elements)
        self.state_element_idx = dict(self.process.state_element_idx)

        # transitions that are specific to this batch, not the process generally:
        self.batch_transitions = {}

    @property
    def measurable_state(self) -> str:
        return self.process.measurable_state

    def F(self) -> Tensor:
        # fill in template:
        F = torch.zeros(size=(self.batch_size, len(self.state_elements), len(self.state_elements)))
        for to_el, from_els in self.transitions.items():
            for from_el, value in from_els.items():
                r, c = self.state_element_idx[to_el], self.state_element_idx[from_el]
                F[:, r, c] = value

        # expand for batch:
        return F.expand(self.batch_size, -1, -1)

    def Q(self) -> Tensor:
        # generate covariance:
        cov = self.process.covariance()
        # expand for batch:
        return cov.expand(self.batch_size, -1, -1)

    def set_transition(self, from_element: str, to_element: str, values: Tensor) -> None:
        assert len(values) == 1 or len(values) == self.batch_size
        r, c = self.state_element_idx[from_element], self.state_element_idx[to_element]
        self.F[:, r, c] = values
