import torch
from torch.autograd import Variable

from torch_kalman.design.design_matrix import DesignMatrix
from torch_kalman.design.nn_output import NNOutput
from torch_kalman.utils.torch_utils import quad_form_diag


class F(DesignMatrix):
    def __init__(self, state_elements):
        self.state_elements = state_elements
        self.transitions = {}
        super().__init__()

    @property
    def template(self):
        if self._template is None:
            num_state_elements = len(self.state_elements)
            self._template = Variable(torch.zeros((num_state_elements, num_state_elements)))
            for (row, col), multiplier in self.transitions.items():
                self._template[row, col] = multiplier()
        return self._template

    def register_variables(self):
        self.nn_outputs = []
        state_idx = {state_id: idx for idx, state_id in enumerate(self.state_elements.keys())}

        for state_id, state in self.state_elements.items():
            from_idx = state_idx[state.id]
            for transition_to_id, multiplier in state.transitions.items():
                to_idx = state_idx[transition_to_id]
                self.transitions[(to_idx, from_idx)] = multiplier
                if isinstance(multiplier, NNOutput):
                    multiplier.add_design_mat_idx((to_idx, from_idx))
                    self.nn_outputs.append(multiplier)


class H(DesignMatrix):
    def __init__(self, state_elements, measures):
        self.state_elements = state_elements
        self.measures = measures
        self.state_to_measures = {}
        super().__init__()

    @property
    def template(self):
        if self._template is None:
            num_measures, num_state_elements = len(self.measures), len(self.state_elements)
            self._template = Variable(torch.zeros((num_measures, num_state_elements)))
            for (row, col), multiplier in self.state_to_measures.items():
                self._template[row, col] = multiplier()
        return self._template

    def register_variables(self):
        self.nn_outputs = []

        state_idx = {state_id: idx for idx, state_id in enumerate(self.state_elements.keys())}
        measure_idx = {measure_id: idx for idx, measure_id in enumerate(self.measures.keys())}

        for measure in self.measures.values():
            for state_id, multiplier in measure.state_elements.items():
                this_measure_idx = measure_idx[measure.id]
                this_state_idx = state_idx[state_id]
                self.state_to_measures.update({(this_measure_idx, this_state_idx): multiplier})
                if isinstance(multiplier, NNOutput):
                    multiplier.add_design_mat_idx((this_measure_idx, this_state_idx))
                    self.nn_outputs.append(multiplier)


class CovarianceMatrix(DesignMatrix):
    def __init__(self):
        self.std_devs = []
        self.corrs = []
        super().__init__()

    def register_variables(self):
        raise NotImplementedError()

    @property
    def num_vars(self):
        raise NotImplementedError()

    @property
    def template(self):
        if self._template is None:
            # populate std-dev:
            diag = Variable(torch.zeros(self.num_vars))
            for idx, std_dev in self.std_devs:
                diag[idx] = std_dev()

            # populate corrs:
            corr_mat = Variable(torch.eye(self.num_vars))
            for (row, col), corr in self.corrs:
                corr_mat[row, col] = corr()

            # create covariance:
            self._template = quad_form_diag(diag, corr_mat)

        return self._template

    def register_covariance(self, variables):
        """
        Take a list of state_elements or measures, and generate a covariance matrix.

        :param variables: A list of state_elements or measures.
        :return: A covariance Matrix, as a pytorch Variable.
        """

        idx_per_var = {var.id: idx for idx, var in enumerate(variables)}
        visited_corr_idx = set()
        for var in variables:
            idx = idx_per_var[var.id]
            if isinstance(var.std_dev(), NNOutput):
                raise NotImplementedError("NNOutputs not currently supported for std-deviations.")
            self.std_devs.append((idx, var.std_dev))
            for id, corr in var.correlations.items():
                if isinstance(corr, NNOutput):
                    raise NotImplementedError("NNOutputs not currently supported for correlations.")
                idx2 = idx_per_var[id]
                if (idx, idx2) not in visited_corr_idx:
                    self.corrs.append(((idx, idx2), corr))
                    self.corrs.append(((idx2, idx), corr))
                    visited_corr_idx.add((idx, idx2))


class Q(CovarianceMatrix):
    def __init__(self, state_elements):
        self.state_elements = state_elements
        super().__init__()

    @property
    def num_vars(self):
        return len(self.state_elements)

    def register_variables(self):
        return self.register_covariance(self.state_elements.values())


class R(CovarianceMatrix):
    def __init__(self, measures):
        self.measures = measures
        super().__init__()

    @property
    def num_vars(self):
        return len(self.measures)

    def register_variables(self):
        return self.register_covariance(self.measures.values())


class B(DesignMatrix):
    def __init__(self):
        raise NotImplementedError()
        # TODO: very similar to H