from typing import Generator, Sequence, Union, Set, Callable, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.process.for_batch import ProcessForBatch


class Process:
    """
    - local level : transition matmul w/ logit link
    - discrete season : transition completely overridden
    - linear model / nn / static fourier : measure matmul w/identity link
    - dynamic fourier : transition matmul w/ identity link
    - offer fx : logit link?

    So the odd one outfe is discrete seasons.
    """

    def __init__(self,
                 id: str,
                 state_elements: Sequence[str]):
        self.id: str = str(id)
        self.state_elements: Sequence[str] = state_elements
        assert len(state_elements) == len(set(state_elements)), "Duplicate `state_elements`."

        self.ses_to_measures = {}
        self.ses_to_measures_ilinks = {}
        self.transitions = {}
        self.transitions_ilinks = {}
        """
        process(for batch)
        1. base-value / intercept (not applicable for var-adjust)
        2. lists of els, where each element is either (a) tensor (if time-invariant), (b) list-of-tensors (if time-varying)
        3. separate dict for (a) elements and (b) elements; each key'd by position in matrix
        4. dict for inv_link function, key'd by position in matrix
        design(for batch) 
        5. create a "base" matrix (all (a)s) on the unconstrained scale.
        6. when asking for matrix at time T, first assign any (b)s, then loop thru to convert to unconstrained scale
        
        other tweaks:
        - ProcessForBatch *_assignments uses idx_in_design to offset the assignments
        """

        self._device = None
        self._state_element_idx = None

        #
        self.expected_batch_kwargs = []

    # measures ---
    def add_measure(self, measure: str):
        # inheritors should usually call `set_measure`, as there's typically a clear choice of which element(s) are measured
        pass

    def _set_measure(self,
                     measure: str,
                     state_element: str,
                     value: Union[float, None],
                     inv_link: Optional[Callable] = None):
        """
        sets the baseline contribution of state_element to measure; establishes a link function for how `adjust_measure`
        changes this baseline.
        passing value=None can aid clarity if the baseline is 0, but this will necessarily be modified in batch (so
        differentiates from an unmeasureable state-element, which is also zero)
        """

        if value is not None:
            assert isinstance(value, float)
            value = torch.Tensor([value])

        key = (measure, state_element)
        assert key not in self.ses_to_measures, f"{key} already set"

        self.ses_to_measures[key] = value
        self.ses_to_measures_ilinks[key] = inv_link

    @property
    def measures(self) -> Set[str]:
        return set(self.ses_to_measures.keys())

    # transitions ---
    def _set_transition(self,
                        from_element: str,
                        to_element: str,
                        value: Union[float, None],
                        inv_link: Optional[Callable] = torch.sigmoid
                        ):

        if value is not None:
            assert isinstance(value, float)
            value = torch.Tensor([value])

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
        raise NotImplementedError

    # other -----
    def parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        raise NotImplementedError

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

    def requires_grad_(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad_(requires_grad=requires_grad)

    def initial_state_means_for_batch(self, parameters: Parameter, num_groups: int, **kwargs) -> Tensor:
        return parameters.expand(num_groups, -1)

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'ProcessForBatch':
        assert self.measures, f"The process `{self.id}` has no measures."
        assert self.transitions, f"The process `{self.id}` has no transitions."
        return ProcessForBatch(process=self,
                               num_groups=num_groups,
                               num_timesteps=num_timesteps)
