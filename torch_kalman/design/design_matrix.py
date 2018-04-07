import torch
from torch.autograd import Variable

from torch_kalman.design import NNOutput
from torch_kalman.design.nn_output import NNOutputTracker
from torch_kalman.utils.torch_utils import quad_form_diag


class DesignMatrix(NNOutputTracker):
    def rebuild_template(self):
        raise NotImplementedError()

    def register_variables(self):
        raise NotImplementedError()

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


class F(DesignMatrix):
    def __init__(self, states, nn_module=None):
        self.states = states
        super().__init__(nn_module=nn_module)

    def rebuild_template(self):
        num_states = len(self.states)
        self._template = Variable(torch.zeros((num_states, num_states)))

    def register_variables(self):
        nn_output_idx = {}

        idx_per_state = {state_id: idx for idx, state_id in enumerate(self.states.keys())}
        for state_id, state in self.states.items():
            from_idx = idx_per_state[state.id]
            for transition_to_id, multiplier in state.transitions.items():
                to_idx = idx_per_state[transition_to_id]
                self._template[to_idx, from_idx] = multiplier()
                if isinstance(multiplier, NNOutput):
                    nn_output_idx[(state.id, transition_to_id)] = (to_idx, from_idx)

        return nn_output_idx


class H(DesignMatrix):
    def __init__(self, states, measurements, nn_module=None):
        self.states = states
        self.measurements = measurements
        super().__init__(nn_module=nn_module)

    def rebuild_template(self):
        num_measurements, num_states = len(self.measurements), len(self.states)
        self._template = Variable(torch.zeros((num_measurements, num_states)))

    def register_variables(self):
        nn_output_idx = {}

        idx_per_measure = {measure_id: idx for idx, measure_id in enumerate(self.measurements.keys())}
        idx_per_state = {state_id: idx for idx, state_id in enumerate(self.states.keys())}

        for measurement in self.measurements.values():
            for state_id, multiplier in measurement.states.items():
                measure_idx = idx_per_measure[measurement.id]
                state_idx = idx_per_state[state_id]
                self._template[measure_idx, state_idx] = multiplier()
                if isinstance(multiplier, NNOutput):
                    nn_output_idx[(measurement.id, state_id)] = (measure_idx, state_idx)

        return nn_output_idx


class Q(DesignMatrix):
    def __init__(self, states, nn_module=None):
        if nn_module:
            raise NotImplementedError("Currently `nn_module` not supported for Q-design-matrix.")
        self.states = states
        super().__init__(nn_module=nn_module)

    def rebuild_template(self):
        self._template = self.covariance_mat_from_variables(self.states.values())

    def register_variables(self):
        return None


class R(DesignMatrix):
    def __init__(self, measurements, nn_module=None):
        if nn_module:
            raise NotImplementedError("Currently `nn_module` not supported for R-design-matrix.")
        self.measurements = measurements
        super().__init__(nn_module=nn_module)

    def rebuild_template(self):
        self._template = self.covariance_mat_from_variables(self.measurements.values())

    def register_variables(self):
        return None


class B(DesignMatrix):
    # TODO: re-read sections on this to make sure you understand it. then its very similar to H
    pass
