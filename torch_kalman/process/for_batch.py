from typing import Union, Tuple, Sequence, Dict

from torch import Tensor

from torch_kalman.design_matrix import TensorOverTime


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

        # transitions:
        self._transitions = None

        # state-element-measurements:
        self._state_measurements = None

    @property
    def transitions(self) -> Dict:
        if self._transitions is None:
            transitions = {}
            for to_el, from_els in {**self.process.transitions, **self.batch_transitions}.items():
                for from_el, values in from_els.items():
                    r, c = self.process.state_element_idx[to_el], self.process.state_element_idx[from_el]
                    if values is None:
                        raise ValueError(f"The value for transition from '{from_el}' to '{to_el}' is None, which means that "
                                         f"this needs to be set on a per-batch basis using the `set_transition` method.")
                    transitions[(r, c)] = values
            self._transitions = transitions
        return self._transitions

    @property
    def state_measurements(self) -> Dict:
        if self._state_measurements is None:
            ses_to_measures = {**self.process.state_elements_to_measures, **self.batch_ses_to_measures}

            state_measurements = {}
            for (measure_id, state_element), values in ses_to_measures.items():
                c = self.state_element_idx[state_element]
                if values is None:
                    raise ValueError(f"The measurement value for measure '{measure_id}' of process '{self.id}' is "
                                     f"None, which means that this needs to be set on a per-batch basis using the "
                                     f"`add_measure` method.")
                state_measurements[(measure_id, c)] = values

            self._state_measurements = state_measurements
        return self._state_measurements

    def set_transition(self, from_element: str, to_element: str, values: Union[Sequence, Tensor]) -> None:
        self._transitions = None

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

        # TODO: don't need tensor-over-time if values is a Tensor w/o requires_grad
        self.batch_transitions[to_element][from_element] = TensorOverTime(values,
                                                                          num_groups=self.num_groups,
                                                                          num_timesteps=self.num_timesteps)

    def add_measure(self,
                    measure: str,
                    state_element: str,
                    values: Union[Sequence, Tensor]) -> None:
        self._state_measurements = None

        assert state_element in self.process.state_elements, f"'{state_element}' is not in this process.'"

        key = (measure, state_element)

        if self.process.state_elements_to_measures.get(key, None):
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process, cannot modify.")

        if key in self.batch_ses_to_measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this batch-process.")

        # TODO: don't need tensor-over-time if values is a Tensor w/o requires_grad
        self.batch_ses_to_measures[key] = TensorOverTime(values,
                                                         num_groups=self.num_groups,
                                                         num_timesteps=self.num_timesteps)
