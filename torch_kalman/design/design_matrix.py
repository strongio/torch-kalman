from collections import OrderedDict

import torch
from torch.autograd import Variable

from torch_kalman.design import NNOutput
from torch_kalman.utils.torch_utils import expand, quad_form_diag


class DesignMatrix(object):
    def __init__(self, nn_module):
        """
        A design-matrix.

        :param nn_module: (Optional) callable. If provided, this is used at the step when the matrix is replicated for each
        item in a batch. Should take the batch as input and return a Variable, the elements of which will fill any elements
        of the matrix that have placeholder NNOutputs in them.
        """
        self.nn_module = nn_module
        self._template, self.nn_output_idx = self.register_template()
        if self.nn_output_idx and not self.nn_module:
            raise ValueError("Some elements contain NNOutputs, so must pass nn_module that will fill them.")

    @property
    def template(self):
        """
        The matrix itself. Any elements that are placeholder NNOutputs will be set to nan.
        :return: A design-matrix.
        """
        if self._template is None:
            self._template, _ = self.register_template()
        return self._template

    def register_template(self):
        raise NotImplementedError()

    def reset(self):
        """
        Reset the 'template' property so the getter will be forced to recreate the template from scratch. This is needed
        because pytorch by default doesn't retain the graph after a call to "backward".
        :return:
        """
        self._template = None

    @staticmethod
    def covariance_mat_from_variables(variables):
        """
        Take a list of states or measurements, and generate a covariance matrix.

        :param variables: A list of states or measurements.
        :return: A covariance Matrix, as a pytorch Variable.
        """
        num_states = len(variables)
        std_devs = Variable(torch.zeros(num_states))
        corr_mat = Variable(torch.eye(num_states))
        idx_per_var = {var.id: idx for idx, var in enumerate(variables)}
        visited_corr_idx = set()
        for var in variables:
            idx = idx_per_var[var.id]
            if isinstance(var.std_dev(), NNOutput):
                raise NotImplementedError("NNOutputs not currently supported for std-deviations.")
            std_devs[idx] = var.std_dev()
            for id, corr in var.correlations.items():
                if isinstance(corr, NNOutput):
                    raise NotImplementedError("NNOutputs not currently supported for correlations.")
                idx2 = idx_per_var[id]
                if (idx, idx2) not in visited_corr_idx:
                    corr_mat[idx, idx2] = corr()
                    corr_mat[idx2, idx] = corr()
                    visited_corr_idx.add((idx, idx2))

        return quad_form_diag(std_devs, corr_mat)

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


class F(DesignMatrix):
    def __init__(self, states, nn_module=None):
        """
        :param states: A list of states, sorted by their IDs.
        :param nn_module: (Optional) callable such as a nn.module, which takes a batch and returns the elements of the
        design matrix that need to be filled in (because they were NNOutputs).
        """
        self.states = states
        super().__init__(nn_module=nn_module)

    def register_template(self):
        nn_output_idx = {}

        num_states = len(self.states)
        template = Variable(torch.zeros((num_states, num_states)))
        idx_per_state = {state.id: idx for idx, state in enumerate(self.states)}

        for state in self.states:
            from_idx = idx_per_state[state.id]
            for transition_to_id, multiplier in state.transitions.items():
                to_idx = idx_per_state[transition_to_id]
                template[to_idx, from_idx] = multiplier()
                if isinstance(multiplier, NNOutput):
                    nn_output_idx[(state.id, transition_to_id)] = (to_idx, from_idx)

        return template, nn_output_idx


class H(DesignMatrix):
    def __init__(self, states, measurements, nn_module=None):
        self.states = states
        self.measurements = measurements
        super().__init__(nn_module=nn_module)

    def register_template(self):
        nn_output_idx = {}

        num_measurements, num_states = len(self.measurements), len(self.states)
        template = Variable(torch.zeros((num_measurements, num_states)))
        idx_per_obs = {obs.id: idx for idx, obs in enumerate(self.measurements)}
        idx_per_state = {state.id: idx for idx, state in enumerate(self.states)}

        for measurement in self.measurements:
            for state_id, multiplier in measurement.states.items():
                measure_idx = idx_per_obs[measurement.id]
                state_idx = idx_per_state[state_id]
                template[measure_idx, state_idx] = multiplier()
                if isinstance(multiplier, NNOutput):
                    nn_output_idx[(measurement.id, state_id)] = (measure_idx, state_idx)

        return template, nn_output_idx


class Q(DesignMatrix):
    def __init__(self, states, nn_module=None):
        if nn_module:
            raise NotImplementedError("Currently `nn_module` not supported for Q-design-matrix.")
        self.states = states
        super().__init__(nn_module=nn_module)

    def register_template(self):
        return self.covariance_mat_from_variables(self.states), None


class R(DesignMatrix):
    def __init__(self, measurements, nn_module=None):
        if nn_module:
            raise NotImplementedError("Currently `nn_module` not supported for R-design-matrix.")
        self.measurements = measurements
        super().__init__(nn_module=nn_module)

    def register_template(self):
        return self.covariance_mat_from_variables(self.measurements), None


class B(DesignMatrix):
    # TODO: re-read sections on this to make sure you understand it. then its very similar to H
    pass
