from typing import Optional, Dict, Union, Tuple

import torch
from torch import Tensor


class ProcessForBatch:
    def __init__(self, process: 'Process', batch_size: int):
        self.batch_size = batch_size
        self.process = process

        # a bit over-protective: batch process gets these, but they're copies so no one will accidentally modify originals
        self.id = str(self.process.id)
        self.state_elements = list(self.process.state_elements)
        self.state_element_idx = dict(self.process.state_element_idx)

        # transitions that are specific to this batch, not the process generally:
        self.batch_transitions = {}
        self.batch_ses_to_measures = {}

    def F(self) -> Tensor:
        if not self.batch_transitions and self.process.F_base is not None:
            return self.process.F_base.expand(self.batch_size, -1, -1)

        # fill in template:
        F = torch.zeros(size=(self.batch_size, len(self.process.state_elements), len(self.process.state_elements)),
                        device=self.process.device)
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
        length = len(values)
        assert length == 1 or length == self.batch_size
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

    def add_measure(self,
                    measure: str,
                    state_element: str,
                    values: Tensor):

        length = len(values)
        assert length == 1 or length == self.batch_size
        assert state_element in self.process.state_elements, f"'{state_element}' is not in this process.'"

        key = (measure, state_element)

        if self.process.state_elements_to_measures.get(key, None):
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process, cannot modify.")

        if key in self.batch_ses_to_measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this batch-process.")

        self.batch_ses_to_measures[key] = values

    def transitions(self) -> Dict[str, Dict[str, Union[Tensor, float]]]:
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

    def state_elements_to_measures(self) -> Dict[Tuple[str, str], Union[Tensor, float]]:
        # don't need to be as careful as w/self.transitions, since dicts of values, not dicts of dicts
        return {**self.process.state_elements_to_measures, **self.batch_ses_to_measures}
