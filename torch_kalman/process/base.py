from typing import Sequence, Union, Set, Callable, Dict, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.process.for_batch import ProcessForBatch

DesignMatAssignment = Union[float, Tensor, Callable]


class Process:
    def __init__(self,
                 id: str,
                 state_elements: Sequence[str]):
        self.id: str = str(id)
        self.state_elements: Sequence[str] = state_elements
        assert len(state_elements) == len(set(state_elements)), "Duplicate `state_elements`."
        self.state_element_idx: Dict[str, int] = {se: i for i, se in enumerate(self.state_elements)}

        self.ses_to_measures: Dict[Tuple[str, str], DesignMatAssignment] = {}
        self.ses_to_measures_ilinks: Dict[Tuple[str, str], Union[Callable, None]] = {}
        self.transitions: Dict[Tuple[str, str], DesignMatAssignment] = {}
        self.transitions_ilinks: Dict[Tuple[str, str], Union[Callable, None]] = {}

        self._device: torch.device = None

        if not set(self.dynamic_state_elements).isdisjoint(self.fixed_state_elements):
            raise ValueError("Class has been misconfigured -- some fixed state-elements are also dynamic-state-elements.")

        self._allow_single_kwarg = False

    def param_dict(self) -> torch.nn.ParameterDict:
        raise NotImplementedError

    # measures ---
    def add_measure(self, measure: str) -> 'Process':
        """
        Calls 'set_measure' with default state_element, value
        """
        raise NotImplementedError

    def _set_measure(self,
                     measure: str,
                     state_element: str,
                     value: DesignMatAssignment,
                     inv_link: Union[Callable, None, bool] = None):
        """
        sets the baseline contribution of state_element to measure; establishes a link function for how `adjust_measure`
        changes this baseline (inv_link=None means identity link-fun)
        """

        assert state_element in self.state_elements

        value = self._check_design_mat_assignment(value)

        key = (measure, state_element)
        assert key not in self.ses_to_measures, f"{key} already set"

        self.ses_to_measures[key] = value
        self.ses_to_measures_ilinks[key] = inv_link

    @property
    def measures(self) -> Set[str]:
        return set(measure for measure, state_element in self.ses_to_measures.keys())

    # transitions ---
    def _set_transition(self,
                        from_element: str,
                        to_element: str,
                        value: DesignMatAssignment,
                        inv_link: Union[Callable, None, bool] = None
                        ):
        """
       sets the baseline transition of between state_elements; establishes a link function for how `adjust_transition`
       changes this baseline (inv_link=None means identity link-fun)
       """
        assert from_element in self.state_elements
        assert to_element in self.state_elements

        value = self._check_design_mat_assignment(value)

        key = (from_element, to_element)
        assert key not in self.transitions, f"{key} already set"

        self.transitions[key] = value
        self.transitions_ilinks[key] = inv_link

    # process variance ---
    # no set_variance: base handled by design, adjustments forced to be link='log'

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        """
        state elements with process-variance
        """
        return self.state_elements

    @property
    def fixed_state_elements(self) -> Sequence[str]:
        """
        state elements with neither process-variance nor initial-variance -- i.e., they are fixed at their initial mean
        """
        return []

    @staticmethod
    def _check_design_mat_assignment(value: DesignMatAssignment) -> Union[Tensor, Callable]:
        if isinstance(value, float):
            value = torch.Tensor([value])
        elif isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError("Design-mat assignment should have only one-element.")
            if value.grad_fn:
                raise ValueError("If `value` is the result of computations that require_grad, need to wrap those "
                                 "computations in a function and pass that for `value` instead.")
        elif not callable(value):
            raise ValueError("`value` must be float, tensor, or a function that produces a tensor")
        return value

    @property
    def device(self):
        if self._device is None:
            raise RuntimeError("Must call `set_device` first.")
        return self._device

    def set_device(self, device: torch.device) -> None:
        self._device = device
        for param in self.param_dict().values():
            param.data = param.data.to(device)

    def requires_grad_(self, requires_grad: bool):
        for param in self.param_dict().values():
            param.requires_grad_(requires_grad=requires_grad)

    def initial_state_means_for_batch(self,
                                      parameters: Parameter,
                                      num_groups: int,
                                      **kwargs) -> Tensor:
        return parameters.expand(num_groups, -1)

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  **kwargs) -> 'ProcessForBatch':
        assert self.measures, f"The process `{self.id}` has no measures."
        assert self.transitions, f"The process `{self.id}` has no transitions."
        return ProcessForBatch(process=self,
                               num_groups=num_groups,
                               num_timesteps=num_timesteps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id.__repr__()})"
