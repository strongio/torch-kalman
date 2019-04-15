from typing import Union, Sequence, Dict, Tuple, List, Callable, Set

import torch

from torch import Tensor

from torch_kalman.utils import split_flat

"""
Each timepoint has a separate design-mat. To assign to the design-mat, you can either pass:
- a scalar tensor, that will be broadcasted to the number of groups
- a 1d tensor with len == num-groups 
- a sequence of the above, with seq-len == num-timesteps
"""
SeqOfTensors = Union[Tuple[Tensor], List[Tensor]]
DesignMatAdjustment = Union[Tensor, SeqOfTensors]


class ProcessForBatch:
    def __init__(self,
                 process: 'Process',
                 num_groups: int,
                 num_timesteps: int):
        self.process = process
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps

        # transitions:
        self._transition_mat_assignments = None
        self.transition_adjustments = {k: [] for k in self.process.transitions.keys()}

        # state-element-measurements:
        self._measurement_mat_assignments = None
        self.measure_adjustments = {k: [] for k in self.process.ses_to_measures.keys()}

        # variance-modifications:
        self._variance_diag_multi_assignments = None
        self.variance_adjustments = {se: [] for se in self.process.dynamic_state_elements}

    @property
    def state_element_idx(self) -> Dict[str, int]:
        return self.process.state_element_idx

    @property
    def measures(self) -> Set[str]:
        return self.process.measures

    # measures ----
    @property
    def measurement_mat_assignments(self) -> Tuple[Dict, Dict]:
        if self._measurement_mat_assignments is None:
            base_vals = {}
            dynamic_vals = {}
            for key, base_value in self.process.ses_to_measures.items():
                ilink_fun = self.process.ses_to_measures_ilinks[key]

                base, dynamic = self._consolidate_adjustments(adjustments=self.measure_adjustments[key],
                                                              trans=ilink_fun,
                                                              intercept=base_value)
                if base is not None:
                    base_vals[key] = base
                if dynamic is not None:
                    dynamic_vals[key] = dynamic

            self._measurement_mat_assignments = base_vals, dynamic_vals

        return self._measurement_mat_assignments

    def adjust_measure(self,
                       measure: str,
                       state_element: str,
                       adjustment: DesignMatAdjustment,
                       check_slow_grad: bool = True):
        key = (measure, state_element)
        if self.process.ses_to_measures_ilinks[key] is False:
            raise Exception(f"{key} is not adjustable")
        adjustment = self._validate_assignment(adjustment, check_slow_grad)
        self._measurement_mat_assignments = None
        self.measure_adjustments[key].append(adjustment)

    # transitions ----
    @property
    def transition_mat_assignments(self) -> Tuple[Dict, Dict]:
        if self._transition_mat_assignments is None:
            base_transitions = {}
            dynamic_transitions = {}
            for key, base_value in self.process.transitions.items():
                ilink_fun = self.process.transitions_ilinks[key]

                base, dynamic = self._consolidate_adjustments(adjustments=self.transition_adjustments[key],
                                                              trans=ilink_fun,
                                                              intercept=base_value)
                if base is not None:
                    base_transitions[key] = base
                if dynamic is not None:
                    dynamic_transitions[key] = dynamic

            self._transition_mat_assignments = base_transitions, dynamic_transitions
        return self._transition_mat_assignments

    def adjust_transition(self,
                          from_element: str,
                          to_element: str,
                          adjustment: DesignMatAdjustment,
                          check_slow_grad: bool = True):
        key = (from_element, to_element)
        if key not in self.process.transitions.keys():
            raise Exception(f"Can't adjust transition for {key}, b/c base value wasn't set (via `Process._set_transition`).")
        if self.process.transitions_ilinks[key] is False:
            raise Exception(f"{key} is not adjustable")
        adjustment = self._validate_assignment(adjustment, check_slow_grad)
        self._transition_mat_assignments = None
        self.transition_adjustments[key].append(adjustment)

    # covariance ---
    @property
    def variance_diag_multi_assignments(self) -> Tuple[Dict, Dict]:
        if self._variance_diag_multi_assignments is None:
            # simpler than transition_mat_assignments and measurement_mat_assignments b/c don't have to worry about link-fun
            base_adjustments = {}
            dynamic_adjustments = {}
            for state_element in self.process.dynamic_state_elements:
                base, dynamic = self._consolidate_adjustments(adjustments=self.variance_adjustments[state_element],
                                                              trans=torch.exp,
                                                              intercept=torch.zeros(1))
                if base is not None:
                    base_adjustments[state_element] = base
                if dynamic is not None:
                    dynamic_adjustments[state_element] = dynamic

            self._variance_diag_multi_assignments = base_adjustments, dynamic_adjustments
        return self._variance_diag_multi_assignments

    def adjust_variance(self,
                        state_element: str,
                        adjustment: DesignMatAdjustment,
                        check_slow_grad: bool = True):
        self._variance_diag_multi_assignments = None
        adjustment = self._validate_assignment(adjustment, check_slow_grad)
        if state_element in self.variance_adjustments.keys():
            self.variance_adjustments[state_element].append(adjustment)
        else:
            raise KeyError(f"`{state_element}` isn't in the dynamic_state_elements of this process.")

    # misc ----
    def _consolidate_adjustments(self,
                                 adjustments: Sequence[DesignMatAdjustment],
                                 intercept: Union[Tensor, Callable],
                                 trans: Union[Callable, None]
                                 ) -> Tuple[Union[DesignMatAdjustment, None], Union[DesignMatAdjustment, None]]:

        if callable(intercept):
            # re-computes value in for_batch, to avoid "gradient has been cleared" pytorch error
            intercept = intercept()

        base = [intercept]
        dynamic = []
        for adjustment in adjustments:
            if isinstance(adjustment, (list, tuple)):
                dynamic.append(adjustment)
            else:
                base.append(adjustment)

        if trans is not None:
            # if there's a transformation function, then assignment must be all base or all dynamic, cannot be a mix
            if len(dynamic):
                # there's at least one dynamic, so all dynamic
                dynamic = list(dynamic) + list(base)
                base = []
            # else:
            # there's no dynamic, so all base

        if len(dynamic):
            reduced_dynamic = [torch.zeros((self.num_groups,), device=self.process.device) for
                               _ in range(self.num_timesteps)]
            # TODO: this code could be optimized
            for tensor_or_seq in dynamic:
                if isinstance(tensor_or_seq, Sequence):
                    reduced_dynamic = [x + y for x, y in zip(reduced_dynamic, tensor_or_seq)]
                else:
                    reduced_dynamic = [x + tensor_or_seq for x in reduced_dynamic]
            # apply transformation:
            if trans:
                reduced_dynamic = [trans(x) for x in reduced_dynamic]
        else:
            reduced_dynamic = None

        if len(base):
            reduced_base = base[0]
            for el in base[1:]:
                reduced_base = reduced_base + el
            # apply transformation:
            if trans:
                reduced_base = trans(reduced_base)
        else:
            reduced_base = None

        return reduced_base, reduced_dynamic

    def _validate_assignment(self,
                             design_mat_assignment: DesignMatAdjustment,
                             check_slow_grad: bool = True) -> DesignMatAdjustment:
        if isinstance(design_mat_assignment, Tensor):

            if list(design_mat_assignment.shape) == [self.num_groups, self.num_timesteps]:
                if design_mat_assignment.requires_grad:
                    raise ValueError("Cannot use group X time tensor as adjustment, unless it does not `require_grad`. "
                                     "To make adjustments that are group and time specific, pass a list of len(times), each "
                                     "containing a 1D tensor w/ len(groups) (or each containing a scalar tensor).")
                design_mat_assignment = split_flat(design_mat_assignment, dim=1, clone=True)
            else:
                self._check_tens(design_mat_assignment, in_list=False, check_slow_grad=check_slow_grad)

        if isinstance(design_mat_assignment, (list, tuple)):
            assert len(design_mat_assignment) == self.num_timesteps
            [self._check_tens(tens, in_list=True, check_slow_grad=check_slow_grad) for tens in design_mat_assignment]
        else:
            raise ValueError("Expected `design_mat_assignment` be list/tuple or tensor")

        return design_mat_assignment

    def _check_tens(self, tens: Tensor, in_list: bool, check_slow_grad: bool = True) -> None:
        if tens.numel() != 1:
            if list(tens.shape) != [self.num_groups]:
                msg = ("Expected {listof}1D tensor{plural} {each}with length == num_groups.".
                       format(listof='list of ' if in_list else '',
                              plural='s' if in_list else '',
                              each='each ' if in_list else ''))
                raise ValueError(msg)
        if in_list:
            if tens.requires_grad and check_slow_grad:
                avoid_funs = {'CopyBackwards', 'SelectBackward'}
                next_fun = tens.grad_fn.next_functions[0][0]
                if (tens.grad_fn.__class__.__name__ in avoid_funs) or (next_fun.__class__.__name__ in avoid_funs):
                    raise RuntimeError(f"An adjustment made for process `{self.process.id}` appears to have been generated "
                                       f"by first creating a tensor that requires_grad, then splitting it into a list of "
                                       f"tensors, one for each time-point. If this is incorrect, avoid this msg by passing "
                                       f"check_slow_grad=False to the adjustment method. Otherwise, this way of creating "
                                       f"adjustments should be avoided because it leads to a very slow backwards pass; "
                                       f"instead, first make a list of tensors, then do computations that require_grad on "
                                       f"each element of the list.")
