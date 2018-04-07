import torch
from torch import Tensor
from numpy import nan
from torch.autograd import Variable

from torch_kalman.design.state import NNState
from torch_kalman.utils.torch_utils import expand


class NNOutput(object):
    def __init__(self):
        """
        This is simply a placeholder that tells NNOutputTracker that a nn-module output will go here.
        """
        self.nan = Tensor([nan])

    def __call__(self, *args, **kwargs):
        return Variable(self.nan)


class NNOutputTracker(object):
    def __init__(self, nn_module):
        self.nn_module = nn_module
        self._template = None
        self.rebuild_template()
        self.nn_output_idx = self.register_variables()
        if self.nn_output_idx and not self.nn_module:
            raise ValueError("Some elements contain NNOutputs, so must pass nn_module that will fill them.")

    def register_variables(self):
        raise NotImplementedError()

    @staticmethod
    def check_nn_output(nn_output, batch_size):
        if nn_output.data.shape[0] != batch_size:
            raise ValueError("The `nn_module` returns an output whose first dimension size != the batch-size.")
        if len(nn_output.data.shape) > 2:
            raise ValueError("The `nn_module` returns an output with more than two dimensions.")
        elif len(nn_output.data.shape) > 1:
            if nn_output.data.shape[1] > 1:
                raise ValueError("The `nn_module` returns a 2D output where the size of the 2nd dimension is > 1. If"
                                 " there are multiple NNOutputs that need to be filled, your `nn_module` should "
                                 "return a dictionary or list of tuples.")

    @property
    def template(self):
        """
        The matrix itself. Any elements that are placeholder NNOutputs will be set to nan.
        :return: A design-matrix.
        """
        if self._template is None:
            self.rebuild_template()
            self.register_variables()
        return self._template

    def rebuild_template(self):
        raise NotImplementedError()

    def reset(self):
        """
        Reset the 'template' property so the getter will be forced to recreate the template from scratch. This is needed
        because pytorch by default doesn't retain the graph after a call to "backward".
        :return:
        """
        self._template = None

    def create_for_batch(self, batch):
        """
        If nn_module was not provided at init, this simply replicates the template to match the batch dimensions. If
        nn_module was provided, the batch is passed to that. Its output will be used to fill in the NNOutputs.

        :param batch: A batch of input data.
        :return: The design matrix expanded so that the first dimension matches the batch-size. Any NNOutputs will be filled.
        """
        bs = batch.data.shape[0]
        expanded = expand(self.template, bs)
        if self.nn_module:
            nn_output = self.nn_module(batch)
            if isinstance(nn_output, Variable):
                # if it's a variable, must be a single value (for each item in the batch).
                # then *all* NNOutputs will get that same value
                self.check_nn_output(nn_output, bs)
                for key, (row, col) in self.nn_output_idx:
                    expanded[:, row, col] = nn_output
            else:
                # otherwise, should be (an object that's coercible to) a dictionary.
                # they keys determine where each output goes
                nn_output = dict(nn_output)
                for key, (row, col) in self.nn_output_idx:
                    expanded[:, row, col] = nn_output[key]
        return expanded


class InitialState(NNOutputTracker):
    def __init__(self, states, nn_module=None):
        self.states = states
        super().__init__(nn_module=nn_module)

    def rebuild_template(self):
        num_states = len(self.states)
        self._template = Variable(torch.zeros(num_states, 1))

    def register_variables(self):
        nn_output_idx = {}
        for i, (state_id, state) in enumerate(self.states.items()):
            self._template[i] = state.initial_value()
            if isinstance(state.initial_value, NNOutput):
                nn_output_idx[state_id] = i
        return nn_output_idx


class NNStateApply(NNOutputTracker):
    def __init__(self, states, nn_module=None):
        self.states = states
        super().__init__(nn_module=nn_module)

    @property
    def template(self):
        raise NotImplementedError("NNStateApply doesn't have a template.")

    def rebuild_template(self):
        pass

    # noinspection PyMethodOverriding
    def create_for_batch(self, batch, state_mean_prev):
        if self.nn_output_idx:
            raise NotImplementedError("TODO")
        else:
            return state_mean_prev

    def register_variables(self):
        nn_output_idx = {}

        for idx, (state_id, state) in enumerate(self.states.items()):
            if isinstance(state, NNState):
                nn_output_idx[state_id] = idx

        return nn_output_idx
