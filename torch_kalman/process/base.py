from typing import Generator, Sequence, Dict, Union, Tuple, Set

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process.for_batch import ProcessForBatch


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
        self.state_elements_to_measures = {}

        # expected kwargs
        self.expected_batch_kwargs = ()

        # in some cases, these won't be generated at init, but later in link_to_design; if created at init then validate
        self._state_elements = state_elements
        self._transitions = transitions
        if (state_elements is not None) and (transitions is not None):
            self.validate_state_elements(state_elements, transitions)

        # device:
        self._device = None

    @property
    def transitions(self):
        return self._transitions

    @property
    def state_elements(self):
        return self._state_elements

    def measures(self) -> Set[str]:
        return set(measure for measure, _ in self.state_elements_to_measures.keys())

    def set_transition(self,
                       from_el: str,
                       to_el: str,
                       value: Union[float, None]) -> None:
        if from_el in self.transitions.keys():
            if self.transitions[from_el].get(to_el, None):
                raise ValueError(f"The transition from '{from_el}' to '{to_el}' is already set.")
        else:
            self.transitions[from_el] = {}
        self.transitions[from_el][to_el] = value

    def add_measure(self,
                    measure: str,
                    state_element: str,
                    value: Union[float, Tensor, None]) -> None:

        assert state_element in self.state_elements, f"'{state_element}' is not in this process.'"

        key = (measure, state_element)

        if key in self.state_elements_to_measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process.")
        self.state_elements_to_measures[key] = value

    def parameters(self) -> Generator[Parameter, None, None]:
        raise NotImplementedError

    def covariance(self) -> Covariance:
        raise NotImplementedError

    def initial_state(self, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @property
    def state_element_idx(self) -> Dict[str, int]:
        if self._state_element_idx is None:
            self._state_element_idx = {el: i for i, el in enumerate(self.state_elements)}
        return self._state_element_idx

    def for_batch(self, input: Tensor, **kwargs) -> 'ProcessForBatch':
        assert self.measures(), f"The process {self.id} has no measures."
        num_groups, num_timesteps, *_ = input.shape
        return ProcessForBatch(process=self,
                               num_groups=num_groups,
                               num_timesteps=num_timesteps,
                               initial_state=self.initial_state_for_batch(num_groups=num_groups, **kwargs))

    def initial_state_for_batch(self, num_groups: int, **kwargs) -> Tuple[Tensor, Tensor]:
        init_mean, init_cov = self.initial_state()
        init_means = init_mean.view(1, *init_mean.shape).expand(num_groups, *init_mean.shape)
        init_covs = init_cov.view(1, *init_cov.shape).expand(num_groups, *init_cov.shape)
        return init_means, init_covs

    def validate_state_elements(self,
                                state_elements: Sequence[str],
                                transitions: Dict[str, Dict[str, Union[float, None]]]):
        # state elements:
        assert len(state_elements) == len(set(state_elements)), "Duplicate `state_elements` not allowed."

        # transitions:
        for to_el, from_els in transitions.items():
            assert to_el in self.state_elements, f"`{to_el}` is in transitions but not state_elements"
            for from_el in from_els.keys():
                assert to_el in self.state_elements, f"`{from_el}` is in transitions but not state_elements"

    def link_to_design(self, design: 'Design') -> None:
        """
        In addition to what's done here, this method is useful for processes that need to know about the design they're
        nested within.

        :param design: The design this process is embedded in.
        """
        self.set_device(design.device)

    @property
    def device(self):
        if self._device is None:
            raise RuntimeError("Must call `set_device` first.")
        return self._device

    def set_device(self, device: torch.device) -> None:
        self._device = device
        for param in self.parameters():
            param.data = param.data.to(device)

    def requires_grad_(self, requires_grad):
        for param in self.parameters():
            param.requires_grad_(requires_grad=requires_grad)

    @property
    def requires_grad(self):
        return any(param.requires_grad for param in self.parameters())

    def set_to_simulation_mode(self, *args, **kwargs):
        """
        Set initial parameters to reasonable values s.t. generating data from this process in a simulation will be reasonable
        """
        self.requires_grad_(False)
