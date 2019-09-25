import functools
import inspect
from copy import copy
from typing import Sequence, Callable, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.process.utils.design_matrix import (
    TransitionMatrix, MeasureMatrix, VarianceMultiplierMatrix, DesignMatAssignment,
    DesignMatAdjustment)


class Process:
    _for_batch = False  # default overridden in for_batch

    def __init__(self, id: str, state_elements: Sequence[str]):
        self.id = str(id)
        self.state_elements = state_elements

        self.transition_mat = TransitionMatrix(self.id, self.state_elements, self.state_elements)
        self.measure_mat = MeasureMatrix(self.id, dim2_names=self.state_elements)
        # TODO: variance-modification is rare/advanced. move to mixin?
        self.variance_multi_mat = VarianceMultiplierMatrix(self.id, self.dynamic_state_elements)

        self._validate()

    def for_batch(self, num_groups: int, num_timesteps: int) -> 'Process':
        if not self.measures:
            raise TypeError(f"The process `{self.id}` has no measures.")
        if self.transition_mat.empty:
            raise TypeError(f"The process `{self.id}` has no transitions.")
        for_batch = copy(self)
        for_batch._for_batch = True
        for_batch.variance_multi_mat = self.variance_multi_mat.for_batch()
        for_batch.measure_mat = self.measure_mat.for_batch()
        for_batch.transition_mat = self.transition_mat.for_batch()
        return for_batch

    @property
    def measures(self):
        return self.measure_mat.measures

    def requires_grad_(self, requires_grad: bool):
        for param in self.param_dict().values():
            param.requires_grad_(requires_grad=requires_grad)

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

    def initial_state_means_for_batch(self, parameters: Parameter, num_groups: int) -> Tensor:
        """
        Most children should use default. Handles rearranging of state-means based on for_batch keyword args. E.g. a
        discrete seasonal process w/ a state-element for each season would need to know on which season the batch starts
        """
        return parameters.expand(num_groups, -1)

    # hidden/util methods ----------------
    def __init_subclass__(cls, **kwargs):
        for method_name in ('for_batch', 'initial_state_means_for_batch'):
            method = getattr(cls, method_name)
            if method:
                setattr(cls, method_name, handle_for_batch_kwargs(method))
        super().__init_subclass__(**kwargs)

    def _validate(self):
        if len(self.state_elements) != len(set(self.state_elements)):
            raise ValueError("Duplicate `state_elements`.")
        if not set(self.dynamic_state_elements).isdisjoint(self.fixed_state_elements):
            raise ValueError("Class has been misconfigured: some fixed state-elements are also dynamic-state-elements.")

    def _set_measure(self,
                     measure: str,
                     state_element: str,
                     value: DesignMatAssignment,
                     ilink: Optional[Callable] = None):
        self.measure_mat.assign(measure=measure, state_element=state_element, value=value)
        self.measure_mat.set_ilink(measure=measure, state_element=state_element, ilink=ilink)

    def _adjust_measure(self,
                        measure: str,
                        state_element: str,
                        adjustment: 'DesignMatAdjustment',
                        check_slow_grad: bool = True):
        if not self._for_batch:
            raise ValueError("Cannot _adjust_measure on the base process, must do so on the output of `for_batch()`.")
        self.measure_mat.adjust(measure=measure,
                                state_element=state_element,
                                value=adjustment,
                                check_slow_grad=check_slow_grad)

    def _set_transition(self,
                        from_element: str,
                        to_element: str,
                        value: DesignMatAssignment,
                        ilink: Optional[Callable] = None):
        self.transition_mat.assign(from_element=from_element, to_element=to_element, value=value)
        self.transition_mat.set_ilink(from_element=from_element, to_element=to_element, ilink=ilink)

    def _adjust_transition(self,
                           from_element: str,
                           to_element: str,
                           adjustment: 'DesignMatAdjustment',
                           check_slow_grad: bool = True):
        if not self._for_batch:
            raise ValueError("Cannot _adjust_transition on the base process, must do so on output of `for_batch()`.")
        self.transition_mat.adjust(from_element=from_element,
                                   to_element=to_element,
                                   value=adjustment,
                                   check_slow_grad=check_slow_grad)

    # no _set_variance: base handled by design, adjustments forced to be link='log'
    def _adjust_variance(self,
                         state_element: str,
                         adjustment: 'DesignMatAdjustment',
                         check_slow_grad: bool = True):
        if not self._for_batch:
            raise ValueError("Cannot _adjust_variance on the base process, must do so on output of `for_batch()`.")
        self.variance_multi_mat.adjust(state_element=state_element, value=adjustment, check_slow_grad=check_slow_grad)

    def __repr__(self) -> str:
        return "{}(id={!r})".format(self.__class__.__name__, self.id)


def handle_for_batch_kwargs(method: Callable) -> Callable:
    """
    Decorates a process's method so that it finds the keyword-arguments that were meant for it.

    :param method: The process's `for_batch` method.
    :return: Decorated version.
    """

    if getattr(method, '_handles_for_batch_kwargs', False):
        return method

    excluded = {'self', 'num_groups', 'num_timesteps'}
    for_batch_kwargs = []
    for kwarg in inspect.signature(method.for_batch).parameters:
        if kwarg in excluded:
            continue
        if kwarg == 'kwargs':
            raise ValueError(f"The signature for {method.__name__} should not use `**kwargs`, should instead specify "
                             f"keyword arguments explicitly.")
        for_batch_kwargs.append(kwarg)

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        new_kwargs = {key: kwargs[key] for key in ('num_groups', 'num_timesteps') if key in kwargs}

        for key in for_batch_kwargs:
            specific_key = "{}__{}".format(self.id, key)
            if specific_key in kwargs:
                new_kwargs[key] = kwargs[specific_key]
            elif key in kwargs:
                new_kwargs[key] = kwargs[key]

        return method(self, *args, **new_kwargs)

    wrapped._handles_for_batch_kwargs = True

    return wrapped
