import inspect
from copy import copy
from typing import Sequence, Callable, Optional, Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.internals.batch import Batchable
from torch_kalman.internals.utils import infer_forward_kwargs

from torch_kalman.process.utils.design_matrix import (
    TransitionMatrix, MeasureMatrix, ProcessVarianceMultiplierMatrix
)
from torch_kalman.internals.repr import NiceRepr
from torch_kalman.process.utils.design_matrix.utils import DesignMatAssignment, DesignMatAdjustment


class Process(NiceRepr, Batchable):
    _repr_attrs = ('id',)

    def __init__(self, id: str, state_elements: Sequence[str], initial_state: Optional[torch.nn.Module] = None):
        self.id = str(id)
        self.state_elements = state_elements

        # transitions:
        self.transition_mat = TransitionMatrix(self.state_elements, self.state_elements)

        # state-element -> measure
        # measures will be appended in add_measure, but state-elements need to be known at init
        self.measure_mat = MeasureMatrix(dim1_names=None, dim2_names=self.state_elements)

        # variance of dynamic state elements:
        self.variance_multi_mat = ProcessVarianceMultiplierMatrix(self.state_elements, self.dynamic_state_elements)

        # a callable that predicts the initial state
        self.initial_state = initial_state or InitialState(self.state_elements)

        self._validate()

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'Process':
        if not self.measures:
            raise TypeError(f"The process `{self.id}` has no measures.")
        if self.transition_mat.empty:
            raise TypeError(f"The process `{self.id}` has no transitions.")
        for_batch = copy(self)
        for_batch.batch_info = num_groups, num_timesteps
        for_batch.variance_multi_mat = self.variance_multi_mat.for_batch(num_groups, num_timesteps)
        for_batch.measure_mat = self.measure_mat.for_batch(num_groups, num_timesteps)
        for_batch.transition_mat = self.transition_mat.for_batch(num_groups, num_timesteps)
        return for_batch

    @property
    def measures(self):
        return self.measure_mat.measures

    def param_dict(self) -> torch.nn.ParameterDict:
        """
        Any parameters that should be exposed to the owning nn.Module.
        """
        p = torch.nn.ParameterDict()
        if hasattr(self.initial_state, 'named_parameters'):
            for nm, param in self.initial_state.named_parameters():
                p['initial_state_' + nm.replace('.', '_')] = param
        return p

    # children should implement ----------------
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

    def initial_state_means_for_batch(self, num_groups: int, **kwargs) -> Tensor:
        if 'num_groups' in self.initial_state._forward_kwargs:
            kwargs['num_groups'] = num_groups
        return self.initial_state(**kwargs)

    # For specifying design -----------:
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

    def batch_kwargs(self) -> Iterable[str]:
        if type(self).for_batch.__code__ == Process.for_batch.__code__:
            yield from ()
            return
        excluded = {'self', 'num_groups', 'num_timesteps'}
        for kwarg in inspect.signature(self.for_batch).parameters:
            if kwarg in excluded:
                continue
            if kwarg == 'kwargs':
                raise TypeError(
                    f"Signature of `{type(self).__name__}.for_batch` must define its keyword args explicitly."
                )
            yield kwarg

    def init_mean_kwargs(self) -> Iterable[str]:
        if not hasattr(self.initial_state, '_forward_kwargs'):
            self.initial_state._forward_kwargs = infer_forward_kwargs(self.initial_state)
        return self.initial_state._forward_kwargs


class InitialState(torch.nn.Module):
    _forward_kwargs = ['num_groups']

    def __init__(self, state_elements: Sequence[str]):
        super().__init__()
        self.mean = torch.nn.Parameter(.1 * torch.randn(len(state_elements)))

    def forward(self, num_groups: int) -> torch.Tensor:
        return self.mean.expand(num_groups, -1).clone()
