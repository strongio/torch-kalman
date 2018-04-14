from collections import OrderedDict

import torch
from torch.nn import ModuleList

from torch_kalman.design.nn_output import DynamicState
from torch_kalman.design.design_matrix import F, H, R, Q, B, InitialState

from torch_kalman.utils.torch_utils import expand


class Design(object):
    def __init__(self):
        self.states = {}
        self.measures = {}
        self.nn_modules = dict(transition={},
                               measure={},
                               state={},
                               initial_state={})

        self.additional_modules = ModuleList()

        self.Q, self.F, self.R, self.H = None, None, None, None
        self.InitialState, self.NNState = None, None
        self._state_idx, self._measure_idx = None, None
        self._measurable_states = None

    def add_nn_module(self, type, nn_module, nn_input, known_to_super):
        if self.finalized:
            raise Exception("Can't add nn_module to design, it's been finalized already.")
        if nn_module in self.nn_modules[type].keys():
            raise Exception("This nn-module was already registered for '{}'.".format(type))
        self.nn_modules[type][nn_module] = nn_input
        if not known_to_super:
            self.additional_modules.append(nn_module)

    def add_state(self, state):
        if self.finalized:
            raise Exception("Can't add state to design, it's been finalized already.")
        if state.id in self.states.keys():
            raise Exception("State with the same ID already in design.")
        self.states[state.id] = state

    def add_states(self, states):
        for state in states:
            self.add_state(state)

    def add_measure(self, measure):
        if self.finalized:
            raise Exception("Can't add measure to design, it's been finalized already.")
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
        self.states = OrderedDict((k, self.states[k]) for k in sorted(self.states.keys()))
        [state.torchify() for state in self.states.values()]
        self.measures = OrderedDict((k, self.measures[k]) for k in sorted(self.measures.keys()))
        [measure.torchify() for measure in self.measures.values()]

        # initial-values:
        self.InitialState = InitialState(self.states)
        self.InitialState.add_nn_inputs(self.nn_modules['initial_state'].items())
        self.InitialState.finalize_nn_module()

        # design-mats:
        self.F = F(states=self.states)
        self.F.add_nn_inputs(self.nn_modules['transition'].items())
        self.F.finalize_nn_module()

        self.H = H(states=self.states, measures=self.measures)
        self.H.add_nn_inputs(self.nn_modules['measure'].items())
        self.H.finalize_nn_module()

        self.Q = Q(states=self.states)
        self.Q.finalize_nn_module()  # currently null

        self.R = R(measures=self.measures)
        self.R.finalize_nn_module()  # currently null

        # NNStates:
        self.NNState = DynamicState(self.states)
        self.NNState.add_nn_inputs(self.nn_modules['state'].items())
        self.NNState.finalize_nn_module()

    @property
    def num_measures(self):
        return len(self.measures)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def measurable_states(self):
        assert self.finalized
        if self._measurable_states is None:
            is_observable = (torch.sum(self.H.template, 0) > 0).data.tolist()
            state_keys = list(self.states.keys())
            self._measurable_states = [self.states[state_keys[i]] for i, observable in enumerate(is_observable)
                                       if observable == 1]
        return self._measurable_states

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

        initial_mean_expanded = self.InitialState.create_for_batch(time=0, **kwargs)

        if self.Q.nn_module.isnull:
            initial_cov = self.Q.template
        else:  # at the time of writing, Q can't have an nn-module. if that changes, the above won't work.
            raise NotImplementedError("Please report this error to the package maintainer")
        bs = initial_mean_expanded.data.shape[0]
        initial_cov_expanded = expand(initial_cov, bs)

        return initial_mean_expanded, initial_cov_expanded

    def state_nn_update(self, state_mean, time, **kwargs):
        self.NNState.update_state_mean(state_mean=state_mean, time=time, **kwargs)

    @property
    def state_idx(self):
        assert self.finalized
        if self._state_idx is None:
            self._state_idx = {state_id: idx for idx, state_id in enumerate(self.states.keys())}
        return self._state_idx

    @property
    def measure_idx(self):
        assert self.finalized
        if self._measure_idx is None:
            self._measure_idx = {measure_id: idx for idx, measure_id in enumerate(self.measures.keys())}
        return self._measure_idx
