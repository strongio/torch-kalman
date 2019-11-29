import functools
import inspect
from copy import copy
from typing import Sequence, Callable, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.process.utils.design_matrix import (
    TransitionMatrix, MeasureMatrix, VarianceMultiplierMatrix, DesignMatAssignment, DesignMatAdjustment
)
from torch_kalman.process.utils.for_batch import is_for_batch


class Process:

    def __init__(self, id: str, state_elements: Sequence[str]):
        self.id = str(id)
        self.state_elements = state_elements
        self._for_batch = False

        # transitions:
        self.transition_mat = TransitionMatrix(self.state_elements, self.state_elements)

        # state-element -> measure
        self.measure_mat = MeasureMatrix(self.state_elements, self.state_elements)

        # variance of dynamic state elements:
        self.variance_multi_mat = VarianceMultiplierMatrix(self.state_elements)
        for state_element in self.dynamic_state_elements:
            self.variance_multi_mat.assign(state_element=state_element, value=0.0)
            self.variance_multi_mat.set_ilink(state_element=state_element, ilink=torch.exp)

        self._validate()

    @is_for_batch(False)
    def for_batch(self, num_groups: int, num_timesteps: int) -> 'Process':
        assert num_groups > 0
        assert num_timesteps > 0
        if not self.measures:
            raise TypeError(f"The process `{self.id}` has no measures.")
        if self.transition_mat.empty:
            raise TypeError(f"The process `{self.id}` has no transitions.")
        for_batch = copy(self)
        for_batch._for_batch = (num_groups, num_timesteps)
        for_batch.variance_multi_mat = self.variance_multi_mat.for_batch(num_groups, num_timesteps)
        for_batch.measure_mat = self.measure_mat.for_batch(num_groups, num_timesteps)
        for_batch.transition_mat = self.transition_mat.for_batch(num_groups, num_timesteps)
        return for_batch

    @property
    @is_for_batch(True)
    def num_groups(self) -> int:
        return self._for_batch[0]

    @property
    @is_for_batch(True)
    def num_timesteps(self) -> int:
        return self._for_batch[1]

    @property
    def measures(self):
        return self.measure_mat.measures

    # children should implement ----------------
    def param_dict(self) -> torch.nn.ParameterDict:
        """
        Any parameters that should be exposed to the owning nn.Module.
        """
        raise NotImplementedError

    @is_for_batch(False)
    def add_measure(self, measure: str) -> 'Process':
        """
        Calls '_set_measure' with default state_element, value
        """
        raise NotImplementedError

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        """
        state elements with process-variance. defaults to all
        """
        return self.state_elements

    @property
    def fixed_state_elements(self) -> Sequence[str]:
        """
        state elements with neither process-variance nor initial-variance -- i.e., they are fixed at their initial mean
        """
        return []

    def initial_state_means_for_batch(self, parameters: Parameter, num_groups: int, **kwargs) -> Tensor:
        """
        Most children should use default. Handles rearranging of state-means based on for_batch keyword args. E.g. a
        discrete seasonal process w/ a state-element for each season would need to know on which season the batch starts
        """
        return parameters.expand(num_groups, -1)

    # util methods ----------------
    def _validate(self):
        if len(self.state_elements) != len(set(self.state_elements)):
            raise ValueError("Duplicate `state_elements`.")
        if not set(self.dynamic_state_elements).isdisjoint(self.fixed_state_elements):
            raise ValueError("Class has been misconfigured: some fixed state-elements are also dynamic-state-elements.")

    @is_for_batch(False)
    def _set_measure(self,
                     measure: str,
                     state_element: str,
                     value: DesignMatAssignment,
                     ilink: Optional[Callable] = None,
                     force: bool = False):
        self.measure_mat.assign(measure=measure, state_element=state_element, value=value, force=force)
        self.measure_mat.set_ilink(measure=measure, state_element=state_element, ilink=ilink, force=force)

    @is_for_batch(True)
    def _adjust_measure(self,
                        measure: str,
                        state_element: str,
                        adjustment: 'DesignMatAdjustment',
                        check_slow_grad: bool = True):
        self.measure_mat.adjust(measure=measure,
                                state_element=state_element,
                                value=adjustment,
                                check_slow_grad=check_slow_grad)

    @is_for_batch(False)
    def _set_transition(self,
                        from_element: str,
                        to_element: str,
                        value: DesignMatAssignment,
                        ilink: Optional[Callable] = None,
                        force: bool = False):
        self.transition_mat.assign(from_element=from_element, to_element=to_element, value=value, force=force)
        self.transition_mat.set_ilink(from_element=from_element, to_element=to_element, ilink=ilink, force=force)

    @is_for_batch(True)
    def _adjust_transition(self,
                           from_element: str,
                           to_element: str,
                           adjustment: 'DesignMatAdjustment',
                           check_slow_grad: bool = True):
        self.transition_mat.adjust(from_element=from_element,
                                   to_element=to_element,
                                   value=adjustment,
                                   check_slow_grad=check_slow_grad)

    # no _set_variance: base handled by design, adjustments forced to be link='log'
    @is_for_batch(True)
    def _adjust_variance(self,
                         state_element: str,
                         adjustment: 'DesignMatAdjustment',
                         check_slow_grad: bool = True):
        self.variance_multi_mat.adjust(state_element=state_element, value=adjustment, check_slow_grad=check_slow_grad)

    def __repr__(self) -> str:
        return "{}(id={!r})".format(self.__class__.__name__, self.id)
