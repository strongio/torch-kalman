from typing import Generator, Sequence, Dict, Union, Set, Callable, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.process.for_batch import ProcessForBatch


class Process:
    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 transitions: Dict[str, Dict[str, Union[float, Callable, None]]]):
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
        self.state_elements = state_elements
        self.transitions = transitions

        # measures:
        self.state_elements_to_measures = {}

        # expected kwargs
        self.expected_batch_kwargs = []

        # state elements:
        assert len(state_elements) == len(set(state_elements)), "Duplicate `state_elements` not allowed."

        # transitions:
        for to_el, from_els in transitions.items():
            assert to_el in self.state_elements, f"`{to_el}` is in transitions but not state_elements"
            for from_el in from_els.keys():
                assert to_el in self.state_elements, f"`{from_el}` is in transitions but not state_elements"

        # variance-adjustments:
        self.var_adjustments = {}

        #
        self._device = None
        self._measures = None
        self._state_element_idx = None

    def __repr__(self):
        return f"{self.__class__.__name__}(id='{self.id}')"

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        """
        state elements with process-variance
        """
        raise NotImplementedError

    @property
    def state_element_idx(self) -> Dict[str, int]:
        if self._state_element_idx is None:
            self._state_element_idx = {el: i for i, el in enumerate(self.state_elements)}
        return self._state_element_idx

    @property
    def dynamic_state_element_idx(self) -> Sequence[int]:
        return [self.state_element_idx[el] for el in self.dynamic_state_elements]

    @property
    def device(self):
        if self._device is None:
            raise RuntimeError("Must call `set_device` first.")
        return self._device

    @property
    def measures(self) -> Set[str]:
        if self._measures is None:
            self._measures = set(measure for measure, _ in self.state_elements_to_measures.keys())
        return self._measures

    def parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        raise NotImplementedError

    def set_transition(self,
                       from_el: str,
                       to_el: str,
                       value: Union[float, Callable, None]) -> None:
        """
        Set the transition from an element to another. In transitioning, the state is multiplied by `value` each timestep.

        If `value` is None then this indicates the transition will be set on a per-batch basis in the `for_batch` method.

        If `value` is a function then this indicates the function will be called on the `ProcessForBatch` object, and the
        value for that batch should be returned.

        These two options are needed for transitions that change over time, and/or transitions that require_grad. In the
        latter case, the result - that we only compute them at the time of batch-creation -- means that we don't need to
        retain the graph when backward is called.
        """
        if from_el in self.transitions.keys():
            if self.transitions[from_el].get(to_el, None):
                raise ValueError(f"The transition from '{from_el}' to '{to_el}' is already set.")
        else:
            self.transitions[from_el] = {}

        assert isinstance(value, float) or (value is None) or callable(value)

        self.transitions[from_el][to_el] = value

    def add_measure(self,
                    measure: str,
                    state_element: Optional[str] = None,
                    value: Union[float, Callable, None] = None) -> None:
        """
        Set the value that determines how a state-element is converted to a measurement. For example, state_element * 1., or
        state_element * 0. for a hidden state-element.

        If `value` is None then this indicates the value will be set on a per-batch basis in the `for_batch` method.

        If `value` is a function then this indicates the function will be called on the `ProcessForBatch` object, and the
        value for that batch should be returned.

        These two options are needed for measurements that change over time, and/or those that require_grad. In the latter
        case, the result - that we only compute them at the time of batch-creation -- means that we don't need to retain the
        graph when backward is called.
        """
        self._measures = None

        assert state_element in self.state_elements, f"'{state_element}' is not in this process.'"

        key = (measure, state_element)

        if key in self.state_elements_to_measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process.")

        assert isinstance(value, float) or (value is None) or callable(value)

        self.state_elements_to_measures[key] = value

    def add_variance_adjustment(self,
                                state_element: str,
                                value: Union[Callable,None]):
        dynamic_state_elements = list(self.dynamic_state_elements)

        if state_element not in dynamic_state_elements:
            raise ValueError("Variance-adjustments are multiplicative, so only apply to elements with process-variance.")

        assert state_element not in self.var_adjustments.keys(), f"Variance-adjustment to '{state_element}' already added."

        if value is not None and not callable(value):
            raise ValueError("`value` must be None, or a callable that will be applied to the ProcessForBatch.")

        self.var_adjustments[state_element] = value

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'ProcessForBatch':
        assert self.measures, f"The process `{self.id}` has no measures."
        assert self.transitions, f"The process `{self.id}` has no transitions."

        return ProcessForBatch(process=self,
                               num_groups=num_groups,
                               num_timesteps=num_timesteps)

    def link_to_design(self, design: 'Design') -> None:
        """
        In addition to what's done here, this method is useful for processes that need to know about the design they're
        nested within.

        :param design: The design this process is embedded in.
        """
        self.set_device(design.device)

    def set_device(self, device: torch.device) -> None:
        self._device = device
        for param in self.parameters():
            param.data = param.data.to(device)

    def requires_grad_(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad_(requires_grad=requires_grad)

    def initial_state_means_for_batch(self, parameters: Parameter, num_groups: int, **kwargs) -> Tensor:
        return parameters.expand(num_groups, -1)
