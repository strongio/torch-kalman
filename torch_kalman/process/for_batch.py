from typing import Union, Sequence, Dict, Tuple, List, Hashable, Any

from torch import Tensor

if False:
    from torch_kalman.process import Process  # for type-hinting w/o circular ref


class TensorAssignment:
    def __init__(self, key: Hashable, value: Any):
        self.key = key
        self._value = value
        self._idx = None

    def set_idx(self, *args):
        self._idx = args

    def get_value(self, **kwargs) -> Tensor:
        return self._value

    @property
    def idx(self) -> Tuple[int, ...]:
        if self._idx is None:
            raise RuntimeError("Must call `set_idx` first.")
        return self._idx


class DynamicTensorAssignment(TensorAssignment):
    pass


class PerTimeTensorAssignment(DynamicTensorAssignment):
    def get_value(self, time: int):
        return self._value[time]


class ProcessForBatch:
    def __init__(self,
                 process: 'Process',
                 num_groups: int,
                 num_timesteps: int):
        self.process = process
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps

        self.id = self.process.id

        # transitions:
        self._transition_mat_assignments = None
        self.transition_adjustments = {k: [] for k in self.process.transitions.keys()}

        # state-element-measurements:
        self._measurement_mat_assignments = None
        self.measure_adjustments = {k: [] for k in self.process.ses_to_measures.keys()}

        # variance-modifications:
        self._variance_diag_multi_assignments = None
        self.variance_adjustments = {se: [] for se in self.process.dynamic_state_elements}

    # measures ----
    @property
    def measurement_mat_assignments(self) -> Tuple[Dict, Dict]:
        if self._measurement_mat_assignments is None:
            base_vals = {}
            dynamic_vals = {}
            for s2m_key, base_value in self.process.ses_to_measures.items():
                ilink_fun = self.process.ses_to_measures_ilinks[s2m_key]
                if ilink_fun is None:
                    if base_value is None:
                        base_value = Tensor([0.])
                    base_vals[s2m_key] = [base_value]
                    dynamic_vals[s2m_key] = []
                    for adjustment in self.measure_adjustments[s2m_key]:
                        if isinstance(adjustment, (list, tuple)):
                            dynamic_vals[s2m_key].append(adjustment)
                        else:
                            base_vals[s2m_key].append(adjustment)
            self._measurement_mat_assignments = base_vals, dynamic_vals

        return self._measurement_mat_assignments

    def adjust_measure(self, measure: str, state_element: str, values: Union[Sequence, Tensor]):
        self._measurement_mat_assignments = None
        self._check_values(values)
        self.measure_adjustments[(measure, state_element)].append(values)

    # transitions ----
    @property
    def transition_mat_assignments(self) -> Tuple[Dict, Dict]:
        if self._transition_mat_assignments is None:
            transitions = []
            for trans_key, base_value in self.process.transitions:
                ilink_fun = self.process.transitions_ilinks[trans_key]
                if base_value is None:
                    base_value = Tensor([0.])

                base_assign = TensorAssignment(key=trans_key, value=base_value)
                transitions.append(base_assign)

                for adjustment in self.transition_adjustments[trans_key]:
                    if isinstance(adjustment, (list, tuple)):
                        adjustment_assign = DynamicTensorAssignment(key=trans_key, value=adjustment)
                    else:
                        adjustment_assign = DynamicTensorAssignment(key=trans_key, value=adjustment)
                    transitions.append(adjustment_assign)

                if ilink_fun is not None:
                    # if there's a link function:
                    # if any dynamic, convert all base to dynamic
                    # if no dynamic ?
                    raise NotImplementedError

            self._transition_mat_assignments = transitions
        return self._transition_mat_assignments

    def adjust_transition(self, from_element: str, to_element: str, values: Union[Sequence, Tensor]):
        self._transition_mat_assignments = None
        self._check_values(values)
        self.transition_adjustments[(from_element, to_element)].append(values)

    # covariance ---
    @property
    def variance_diag_multi_assignments(self) -> Dict:
        raise NotImplementedError

    def adjust_variance(self, state_element: str, values: Union[Sequence, Tensor]):
        self._variance_diag_multi_assignments = None
        self._check_values(values)
        self.variance_adjustments[state_element].append(values)

    # misc ----
    def _check_values(self, values: Union[Tensor, Tuple, List]) -> None:
        if isinstance(values, Tensor):
            self._check_tens(values, in_list=False)
        elif isinstance(values, (list, tuple)):
            assert len(values) == self.num_timesteps
            [self._check_tens(tens, in_list=True) for tens in values]
        else:
            raise ValueError("Expected `values` be list/tuple or tensor")

    def _check_tens(self, tens: Tensor, in_list: bool):
        if tens.numel() != 1:
            if list(tens.shape) != [self.num_groups]:
                msg = ("Expected {listof}1D tensor{plural} {each}with length == num_groups.".
                       format(listof='list of ' if in_list else '',
                              plural='s' if in_list else '',
                              each='each ' if in_list else ''))
                raise ValueError(msg)
        if in_list:
            if tens.requires_grad and tens.grad_fn.__class__.__name__ == 'CopyBackwards':
                raise RuntimeError("Please report this error to the package maintainer.")
