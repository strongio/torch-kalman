import inspect
from copy import copy
from typing import Sequence, Callable, Optional, Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.batch import Batchable
from torch_kalman.process.utils.design_matrix import (
    TransitionMatrix, MeasureMatrix, VarianceMultiplierMatrix, DesignMatAssignment, DesignMatAdjustment
)


class Process(Batchable):

    def __init__(self, id: str, state_elements: Sequence[str]):
        self.id = str(id)
        self.state_elements = state_elements

        # transitions:
        self.transition_mat = TransitionMatrix(self.state_elements, self.state_elements)

        # state-element -> measure
        # measures will be appended in add_measure, but state-elements need to be known at init
        self.measure_mat = MeasureMatrix(dim1_names=None, dim2_names=self.state_elements)

        # variance of dynamic state elements:
        self.variance_multi_mat = VarianceMultiplierMatrix(self.state_elements)

        self._validate()

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'Process':
        if self._batch_info:
            raise RuntimeError("Cannot call `for_batch()` on output of `for_batch()`.")
        if not self.measures:
            raise TypeError(f"The process `{self.id}` has no measures.")
        if self.transition_mat.empty:
            raise TypeError(f"The process `{self.id}` has no transitions.")
        for_batch = copy(self)
        for_batch._batch_info = num_groups, num_timesteps
        for_batch.variance_multi_mat = self.variance_multi_mat.for_batch(num_groups, num_timesteps)
        for_batch.measure_mat = self.measure_mat.for_batch(num_groups, num_timesteps)
        for_batch.transition_mat = self.transition_mat.for_batch(num_groups, num_timesteps)
        return for_batch

    @property
    def measures(self):
        return self.measure_mat.measures

    # children should implement ----------------
    def param_dict(self) -> torch.nn.ParameterDict:
        """
        Any parameters that should be exposed to the owning nn.Module.
        """
        raise NotImplementedError

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

    # TODO -----------:
    def _set_measure(self,
                     measure: str,
                     state_element: str,
                     value: DesignMatAssignment,
                     ilink: Optional[Callable] = None,
                     force: bool = False):
        self.measure_mat.assign(measure=measure, state_element=state_element, value=value, overwrite=force)
        self.measure_mat.set_ilink(measure=measure, state_element=state_element, ilink=ilink, overwrite=force)

    def _adjust_measure(self,
                        measure: str,
                        state_element: str,
                        adjustment: 'DesignMatAdjustment',
                        check_slow_grad: bool = True):
        self.measure_mat.adjust(
            measure=measure,
            state_element=state_element,
            value=adjustment,
            check_slow_grad=check_slow_grad
        )

    def _set_transition(self,
                        from_element: str,
                        to_element: str,
                        value: DesignMatAssignment,
                        ilink: Optional[Callable] = None,
                        force: bool = False):
        self.transition_mat.assign(from_element=from_element, to_element=to_element, value=value, overwrite=force)
        self.transition_mat.set_ilink(from_element=from_element, to_element=to_element, ilink=ilink, overwrite=force)

    def _adjust_transition(self,
                           from_element: str,
                           to_element: str,
                           adjustment: 'DesignMatAdjustment',
                           check_slow_grad: bool = True):
        self.transition_mat.adjust(
            from_element=from_element,
            to_element=to_element,
            value=adjustment,
            check_slow_grad=check_slow_grad
        )

    # no _set_variance: base handled by design, adjustments forced to be link='log'
    def _adjust_variance(self,
                         state_element: str,
                         adjustment: 'DesignMatAdjustment',
                         check_slow_grad: bool = True):
        self.variance_multi_mat.adjust(state_element=state_element, value=adjustment, check_slow_grad=check_slow_grad)

    # util methods ----------------
    def _validate(self):
        if len(self.state_elements) != len(set(self.state_elements)):
            raise ValueError("Duplicate `state_elements`.")
        if not set(self.dynamic_state_elements).isdisjoint(self.fixed_state_elements):
            raise ValueError("Class has been misconfigured: some fixed state-elements are also dynamic-state-elements.")

    def __init_subclass__(cls, **kwargs):
        for_batch_kwargs = set(cls.for_batch_kwargs(cls.for_batch))
        init_mean_kwargs = set(cls.for_batch_kwargs(cls.initial_state_means_for_batch))

        overrides_for_batch = (cls.for_batch != Process.for_batch)
        overrides_init_mean = (cls.initial_state_means_for_batch != Process.initial_state_means_for_batch)

        if overrides_for_batch:
            if 'kwargs' in for_batch_kwargs:
                raise TypeError(
                    f"Signature of `{cls.__name__}.for_batch` must define its keyword args explicitly."
                )
        if overrides_init_mean:
            if 'kwargs' in init_mean_kwargs:
                raise TypeError(
                    f"Signature of `{cls.__name__}.initial_state_means_for_batch` must define keyword args explicitly."
                )
        if overrides_for_batch and overrides_init_mean:
            if for_batch_kwargs != init_mean_kwargs:
                raise TypeError(
                    f"`{cls.__name__}.initial_state_means_for_batch()` must match signature of .for_batch()"
                )
        super().__init_subclass__()

    @classmethod
    def for_batch_kwargs(cls, method=None) -> Iterable[str]:
        if method is None:
            method = cls.for_batch
        excluded = {'self', 'num_groups', 'num_timesteps', 'parameters'}
        for kwarg in inspect.signature(method).parameters:
            if kwarg in excluded:
                continue
            yield kwarg

    def __repr__(self) -> str:
        return "{}(id={!r})".format(type(self).__name__, self.id)
