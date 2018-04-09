from collections import OrderedDict, defaultdict

import torch
from torch.autograd import Variable

from torch_kalman.design.nn_output import NNOutput, DynamicState
from torch_kalman.design.design_matrix import F, H, R, Q, B, InitialState

from torch_kalman.utils.torch_utils import expand

from IPython.core.debugger import Pdb

pdb = Pdb()


class Design(object):
    def __init__(self):
        self.states = {}
        self.measurements = {}
        self.nn_modules = dict(transition={},
                               measurement={},
                               state={},
                               initial_state={})

        self.Q, self.F, self.R, self.H = None, None, None, None
        self.InitialState, self.NNState = None, None

    def add_nn_module(self, nn_module, type, input_name):
        if self.finalized:
            raise Exception("Can't add nn_module to design, it's been finalized already.")
        if nn_module in self.nn_modules[type].keys():
            raise Exception("This nn-module was already registered for '{}'.".format(type))
        self.nn_modules[type][nn_module] = input_name

    def add_state(self, state):
        if self.finalized:
            raise Exception("Can't add state to design, it's been finalized already.")
        if state.id in self.states.keys():
            raise Exception("State with the same ID already in design.")
        self.states[state.id] = state

    def add_measurement(self, measurement):
        if self.finalized:
            raise Exception("Can't add measurement to design, it's been finalized already.")
        if measurement.id in self.measurements.keys():
            raise Exception("Measurement with the same ID already in design.")
        self.measurements[measurement.id] = measurement

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

        # organize the states/measurements:
        self.states = OrderedDict((k, self.states[k]) for k in sorted(self.states.keys()))
        [state.torchify() for state in self.states.values()]
        self.measurements = OrderedDict((k, self.measurements[k]) for k in sorted(self.measurements.keys()))
        [measurement.torchify() for measurement in self.measurements.values()]

        # initial-values:
        self.InitialState = InitialState(self.states)
        self.InitialState.add_nn_inputs(self.nn_modules['initial_state'].items())
        self.InitialState.finalize_nn_module()

        # design-mats:
        self.F = F(states=self.states)
        self.F.add_nn_inputs(self.nn_modules['transition'].items())
        self.F.finalize_nn_module()
        self.check_for_input_name_collision(self.F.input_names)

        self.H = H(states=self.states, measurements=self.measurements)
        self.H.add_nn_inputs(self.nn_modules['measurement'].items())
        self.H.finalize_nn_module()
        self.check_for_input_name_collision(self.H.input_names)

        self.Q = Q(states=self.states)
        self.Q.finalize_nn_module()  # currently null

        self.R = R(measurements=self.measurements)
        self.R.finalize_nn_module()  # currently null

        # NNStates:
        self.NNState = DynamicState(self.states)
        self.NNState.add_nn_inputs(self.nn_modules['state'].items())
        self.NNState.finalize_nn_module()
        self.check_for_input_name_collision(self.NNState.input_names)

    def check_for_input_name_collision(self, input_names):
        for input_name in input_names:
            if input_name in self.InitialState.input_names:
                raise Exception("The input_name '{}' was used for an input to the `initial_state` nn_module, so it can't be "
                                "used for other nn_modules.".format(input_name))

    @property
    def num_measurements(self):
        return len(self.measurements)

    @property
    def num_states(self):
        return len(self.states)

    def reset(self):
        """
        Reset the design matrices. For efficiency, the code to generate these matrices isn't executed every time
        they are called; instead, it's executed once then the results are saved. But calling pytorch's backward method
        will clear the graph and so these matrices will need to be re-generated. So this function is called at the end
        of the forward pass computations, so that the matrices will be rebuild next time it's called.
        """
        self.F.reset()
        self.H.reset()
        self.Q.reset()
        self.R.reset()
        self.InitialState.reset()

    def initialize_state(self, **kwargs):

        initial_mean_expanded = self.InitialState.create_for_batch(time=0, **kwargs)

        if self.Q.nn_module is None:
            initial_cov = self.Q.template
        else:  # at the time of writing, Q can't have an nn-module. if that changes, the above won't work.
            raise NotImplementedError("Please report this error to the package maintainer")
        bs = initial_mean_expanded.data.shape[0]
        initial_cov_expanded = expand(initial_cov, bs)

        return initial_mean_expanded, initial_cov_expanded

    def state_nn_update(self, state_mean, time, **kwargs):
        raise NotImplementedError()
