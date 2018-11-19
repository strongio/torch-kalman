from typing import Optional, Dict, Union, Tuple

import torch
from torch import Tensor


class ProcessForBatch:
    def __init__(self,
                 process: 'Process',
                 num_groups: int,
                 num_timesteps: int,
                 initial_state: Tuple[Tensor, Tensor]):

        self.process = process
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps

        self.id = str(self.process.id)
        self.state_elements = list(self.process.state_elements)
        self.state_element_idx = dict(self.process.state_element_idx)

        # transitions that are specific to this batch, not the process generally:
        self.batch_transitions = {}
        self.batch_ses_to_measures = {}

        # initial state:
        self.initial_state = initial_state

    def set_transition(self, from_element: str, to_element: str, values: Tensor) -> None:
        if values.numel() != 1:
            assert list(values.shape) == [self.num_groups, self.num_timesteps]
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

        if values.numel() != 1:
            assert list(values.shape) == [self.num_groups, self.num_timesteps]
        assert state_element in self.process.state_elements, f"'{state_element}' is not in this process.'"

        key = (measure, state_element)

        if self.process.state_elements_to_measures.get(key, None):
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process, cannot modify.")

        if key in self.batch_ses_to_measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this batch-process.")

        self.batch_ses_to_measures[key] = values

    def state_elements_to_measures(self) -> Dict[Tuple[str, str], Union[Tensor, float]]:
        return {**self.process.state_elements_to_measures, **self.batch_ses_to_measures}

    def F(self) -> Tensor:
        F = torch.zeros(self.num_groups, self.num_timesteps, len(self.state_elements), len(self.state_elements))
        for to_el, from_els in {**self.process.transitions, **self.batch_transitions}.items():
            for from_el, values in from_els.items():
                r, c = self.process.state_element_idx[to_el], self.process.state_element_idx[from_el]
                if values is None:
                    raise ValueError(f"The value for transition from '{from_el}' to '{to_el}' is None, which means that this"
                                     f" needs to be set on a per-batch basis using the `set_transition` method.")
                F[:, :, r, c] = values
        return F

    def Q(self) -> Tensor:
        cov = self.process.covariance()
        return cov.view(1, 1, *cov.shape).expand(self.num_groups, self.num_timesteps, -1, -1)
