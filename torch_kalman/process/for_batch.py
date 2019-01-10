from typing import Union, Sequence, Dict, Tuple, List

from torch import Tensor

if False:
    from torch_kalman.process import Process  # for type-hinting w/o circular ref


class ProcessForBatch:
    def __init__(self,
                 process: 'Process',
                 num_groups: int,
                 num_timesteps: int):

        self.process = process
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps

        self.id = str(self.process.id)
        self.state_elements = list(self.process.state_elements)
        self.state_element_idx = dict(self.process.state_element_idx)

        # transitions:
        self._transition_mat_assignments = None
        self.batch_transitions = {}

        # state-element-measurements:
        self._measurement_mat_assignments = None
        self.batch_ses_to_measures = {}

        # variance-modifications:
        self._variance_diag_multi_assignments = None
        self.batch_var_adjustments = {}

    # covariance ---
    @property
    def variance_diag_multi_assignments(self) -> Dict:
        if self._variance_diag_multi_assignments is None:
            base = {}
            # for base-mods, functions get converted to values now
            for k, value in self.process.var_adjustments.items():
                base[k] = value(self) if callable(value) else value

            mods = {}
            for state_element, values in {**base, **self.batch_var_adjustments}.items():
                idx = self.process.state_element_idx[state_element]
                if values is None:
                    raise ValueError(f"The value for variance-modification for '{state_element}' is None, which means that "
                                     f"this needs to be set on a per-batch basis using the `add_variance_mod` method.")
                mods[(idx, idx)] = values
            self._variance_diag_multi_assignments = mods
        return self._variance_diag_multi_assignments

    def add_variance_adjustment(self,
                                state_element: str,
                                values: Union[Sequence, Tensor]):
        self._variance_diag_multi_assignments = None

        assert state_element in self.process.state_elements

        self._check_values(values)

        already = self.process.var_adjustments.get(state_element, None)
        if already:
            raise ValueError(f"The variance for '{state_element}' was already modified for this batch, so do so again.")

        if state_element in self.batch_var_adjustments.keys():
            raise ValueError(f"The var-adjustment for '{state_element}' was already set for this Process, so can't give "
                             f"it batch-specific values (unless set to `None`).")

        # TODO: require None in base?

        self.batch_var_adjustments[state_element] = values

    # transitions ----
    @property
    def transition_mat_assignments(self) -> Dict:
        if self._transition_mat_assignments is None:
            transitions = {}

            # merge transitions, with batch-transitions taking precedence:
            all_transitions = {}
            # for base-transitions, functions get converted to values now
            for to_el, from_els in self.process.transitions.items():
                for from_el, value in from_els.items():
                    if callable(value):
                        value = value(self)
                    all_transitions[(to_el, from_el)] = value

            for to_el, from_els in self.batch_transitions.items():
                for from_el, value in from_els.items():
                    all_transitions[(to_el, from_el)] = value

            # check value, convert to idxs:
            for (to_el, from_el), value in all_transitions.items():
                r, c = self.process.state_element_idx[to_el], self.process.state_element_idx[from_el]
                if value is None:
                    raise ValueError(f"The value for transition from '{from_el}' to '{to_el}' is None, which means that "
                                     f"this needs to be set on a per-batch basis using the `set_transition` method.")
                transitions[(r, c)] = value
            self._transition_mat_assignments = transitions
        return self._transition_mat_assignments

    def set_transition(self, from_element: str, to_element: str, values: Union[Sequence, Tensor]) -> None:
        self._transition_mat_assignments = None

        assert from_element in self.process.state_elements
        assert to_element in self.process.state_elements

        if to_element in self.process.transitions.keys():
            already = self.process.transitions[to_element].get(from_element, None)
            if already:
                raise ValueError(f"The transition from '{from_element}' to '{to_element}' was already set for this Process,"
                                 f" so can't give it batch-specific values (unless set to `None`).")
        else:
            raise ValueError(f"The transition from '{from_element}' to '{to_element}' must be `None` in the process in order"
                             " to set transitions on a per-batch basis; use `Process.set_transition(value=None)`.")

        if to_element not in self.batch_transitions.keys():
            self.batch_transitions[to_element] = {}
        elif from_element in self.batch_transitions[to_element]:
            raise ValueError(f"The transition from '{from_element}' to '{to_element}' was already set for this batch,"
                             f" so can't set it again.")

        self._check_values(values)

        self.batch_transitions[to_element][from_element] = values

    # measures ----
    @property
    def measurement_mat_assignments(self) -> Dict:
        if self._measurement_mat_assignments is None:
            base = {}
            # for base-measurements, functions get converted to values now
            for k, value in self.process.state_elements_to_measures.items():
                base[k] = value(self) if callable(value) else value
            ses_to_measures = {**base, **self.batch_ses_to_measures}

            state_measurements = {}
            for (measure_id, state_element), values in ses_to_measures.items():
                c = self.state_element_idx[state_element]
                if values is None:
                    raise ValueError(f"The measurement value for measure '{measure_id}' of process '{self.id}' is "
                                     f"None, which means that this needs to be set on a per-batch basis using the "
                                     f"`add_measure` method.")
                state_measurements[(measure_id, c)] = values

            self._measurement_mat_assignments = state_measurements
        return self._measurement_mat_assignments

    def add_measure(self,
                    measure: str,
                    state_element: str,
                    values: Union[Sequence, Tensor]) -> None:
        self._measurement_mat_assignments = None

        assert state_element in self.process.state_elements, f"'{state_element}' is not in this process.'"

        key = (measure, state_element)

        already = self.process.state_elements_to_measures.get(key, None)
        if already:
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process, cannot modify.")

        if key in self.batch_ses_to_measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this batch-process.")

        # TODO: require None in base?

        self._check_values(values)
        self.batch_ses_to_measures[key] = values

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
