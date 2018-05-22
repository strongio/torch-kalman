from collections import OrderedDict

import torch
from torch.nn import ModuleList

from torch_kalman.design.variable_tracker.nn_state import NNState
from torch_kalman.design.design_matrix.design_matrices import F, H, Q, R, B
from torch_kalman.design.design_matrix.initial_state import InitialState


class Design(object):
    def __init__(self):
        self.state_elements = dict()
        self.measures = dict()
        self.nn_modules = dict(transition=dict(),
                               measure=dict(),
                               state=dict(),
                               initial_state=dict())

        self.additional_modules = ModuleList()

        self.Q, self.F, self.R, self.H = None, None, None, None
        self.InitialState, self.NNState = None, None
        self._state_idx, self._measure_idx = None, None
        self._measurable_state_elements = None

    def add_nn_module(self, type, nn_module, nn_inputs, known_to_super):
        assert not self.finalized
        assert nn_module not in self.nn_modules[type].keys()
        self.nn_modules[type][nn_module] = nn_inputs
        if not known_to_super:
            self.additional_modules.append(nn_module)

    def add_state_element(self, state):
        assert not self.finalized
        if state.id in self.state_elements.keys():
            raise Exception("State with the same ID already in design.")
        self.state_elements[state.id] = state

    def add_state_elements(self, states):
        for state in states:
            self.add_state_element(state)

    def add_measure(self, measure):
        assert not self.finalized
        if measure.id in self.measures.keys():
            raise Exception("Measurement with the same ID already in design.")
        self.measures[measure.id] = measure

    def add_measures(self, measures):
        for measure in measures:
            self.add_measure(measure)

    def add_process(self, process):
        raise NotImplementedError()

    @property
    def finalized(self):
        finalized_list = [x is not None for x in (self.Q, self.F, self.R, self.H, self.InitialState, self.NNState)]
        if all(finalized_list):
            return True
        if not any(finalized_list):
            return False
        raise Exception("This design is only partially finalized, an error must have occurred.")

    def finalize(self):
        if self.finalized:
            raise Exception("Design was already finalized.")

        # organize the states/measures:
        self.state_elements = OrderedDict((k, self.state_elements[k]) for k in sorted(self.state_elements.keys()))
        [state.torchify() for state in self.state_elements.values()]
        self.measures = OrderedDict((k, self.measures[k]) for k in sorted(self.measures.keys()))
        [measure.torchify() for measure in self.measures.values()]

        # initial-values:
        self.InitialState = InitialState(self.state_elements)
        self.InitialState.prepare_nn_module(self.nn_modules['initial_state'])

        # design-mats:
        self.F = F(self.state_elements)
        self.F.prepare_nn_module(self.nn_modules['transition'])

        self.H = H(self.state_elements, self.measures)
        self.H.prepare_nn_module(self.nn_modules['measure'])

        self.Q = Q(self.state_elements)
        self.Q.prepare_nn_module({})  # currently null

        self.R = R(self.measures)
        self.R.prepare_nn_module({})  # currently null

        # NNStates:
        self.NNState = NNState(self.state_elements)
        self.NNState.prepare_nn_module(self.nn_modules['state'])

    @property
    def num_measures(self):
        return len(self.measures)

    @property
    def num_state_elements(self):
        return len(self.state_elements)

    @property
    def measurable_state_elements(self):
        assert self.finalized
        if self._measurable_state_elements is None:
            is_observable = (torch.sum(self.H.template, 0) > 0).data.tolist()
            state_keys = list(self.state_elements.keys())
            self._measurable_state_elements = [self.state_elements[state_keys[i]] for i, observable in
                                               enumerate(is_observable) if observable == 1]
        return self._measurable_state_elements

    def reset(self):
        """
        Reset the design matrices. For efficiency, the code to generate these matrices isn't executed every time
        they are called; instead, it's executed once then the results are saved. But calling pytorch's backward method
        will clear the graph and so these matrices will need to be re-generated. So this function is called at the end
        of the forward pass computations, so that the matrices will be rebuilt next time it's called.
        """
        self.F.reset()
        self.H.reset()
        self.Q.reset()
        self.R.reset()
        self.InitialState.reset()

    def initialize_state(self, **kwargs):

        initial_mean_expanded = self.InitialState.create_means_for_batch(time=0, **kwargs)
        initial_cov_expanded = self.InitialState.create_cov_for_batch(time=0, **kwargs)

        return initial_mean_expanded, initial_cov_expanded

    def state_nn_update(self, state_mean, time, **kwargs):
        self.NNState.update_state_mean(state_mean=state_mean, time=time, **kwargs)

    @property
    def state_idx(self):
        assert self.finalized
        if self._state_idx is None:
            self._state_idx = {state_id: idx for idx, state_id in enumerate(self.state_elements.keys())}
        return self._state_idx

    @property
    def measure_idx(self):
        assert self.finalized
        if self._measure_idx is None:
            self._measure_idx = {measure_id: idx for idx, measure_id in enumerate(self.measures.keys())}
        return self._measure_idx


def reset_design_on_exit(func):
    def _decorated(self, *args, **kwargs):
        try:
            out = func(self, *args, **kwargs)
            self.design.reset()
            return out
        except:
            self.design.reset()
            raise

    return _decorated
