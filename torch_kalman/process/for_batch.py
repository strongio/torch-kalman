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
        leftover_transitions = self.process.transitions_to_fill.copy()
        # fill in template:
        F = self.process.F_base.expand(self.batch_size, -1, -1)
        for to_el, from_els in self.batch_transitions.items():
            for from_el, values in from_els.items():
                if (to_el, from_el) in leftover_transitions:
                    leftover_transitions.remove((to_el, from_el))
                r, c = self.process.state_element_idx[to_el], self.process.state_element_idx[from_el]
                F[:, r, c] = values

        if leftover_transitions:
            raise ValueError(f"Following transitions need to filled in the batch:\n{leftover_transitions}")

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
                                 f" so can't give it batch-specific values (unless set to `None`).")
        else:
            raise ValueError(f"The transition from '{from_element}' to '{to_element}' must be `None` in the process in order"
                             " to set transitions on a per-batch basis; use `Process.set_transition(value=None)`.")

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

    def state_elements_to_measures(self) -> Dict[Tuple[str, str], Union[Tensor, float]]:
        # don't need to be as careful as w/self.transitions, since dicts of values, not dicts of dicts
        return {**self.process.state_elements_to_measures, **self.batch_ses_to_measures}

    def has_batch_transitions(self):
        return self.process.has_batch_transitions()
