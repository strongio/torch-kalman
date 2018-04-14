import torch
from torch.autograd import Variable

from torch_kalman.design.nn_output import NNOutput, NNOutputTracker
from torch_kalman.utils.torch_utils import quad_form_diag, expand


class DesignMatrix(NNOutputTracker):
    def __init__(self):
        self._template = None
        self.batch_cache = {}
        super().__init__()

    @property
    def template(self):
        raise NotImplementedError()

    def register_variables(self):
        raise NotImplementedError()

    def reset(self):
        # TODO: what if they terminate the forward-pass partway through?
        self._template = None
        self.batch_cache = {}

    def create_for_batch(self, time, **kwargs):
        if time not in self.batch_cache.keys():
            bs = kwargs['kf_input'].data.shape[0]

            if self.nn_module.isnull:
                self.batch_cache[time] = expand(self.template, bs)
            else:
                # check kwargs:
                missing_kwargs = self.input_names - set(kwargs.keys())
                if len(missing_kwargs) == 0:
                    nn_module_kwargs = {argname: kwargs[argname] for argname in self.input_names}
                else:
                    raise TypeError("missing {} required arguments: {}".format(len(missing_kwargs), missing_kwargs))

                # expand, replacing NNOutput placeholders:
                nn_outputs = self.nn_module(time=time, **nn_module_kwargs)
                expanded = expand(self.template, bs).clone()
                for (row, col), output in nn_outputs:
                    expanded[:, row, col] = output
                self.batch_cache[time] = expanded

        return self.batch_cache[time]


class InitialState(DesignMatrix):
    def __init__(self, states):
        """
        Not really a design-matrix, but uses all the same methods.

        :param states: The States.
        """
        self.states = states
        self.initial_states = {}
        super().__init__()

    @property
    def template(self):
        if self._template is None:
            num_states = len(self.states)
            self._template = Variable(torch.zeros(num_states, 1))
            for i, initial_state in self.initial_states.items():
                self._template[i] = initial_state()
        return self._template

    def register_variables(self):
        self.nn_outputs = []

        for i, (state_id, state) in enumerate(self.states.items()):
            self.initial_states.update({i: state.initial_value})
            if isinstance(state.initial_value, NNOutput):
                state.initial_value.add_design_mat_idx((i, 0))
                self.nn_outputs.append(state.initial_value)

    def create_for_batch(self, time, **kwargs):
        if time != 0:
            raise Exception("InitialState is for time-zero only.")

        return super().create_for_batch(time=time, **kwargs)


class F(DesignMatrix):
    def __init__(self, states):
        self.states = states
        self.transitions = {}
        super().__init__()

    @property
    def template(self):
        if self._template is None:
            num_states = len(self.states)
            self._template = Variable(torch.zeros((num_states, num_states)))
            for (row, col), multiplier in self.transitions.items():
                self._template[row, col] = multiplier()
        return self._template

    def register_variables(self):
        self.nn_outputs = []
        state_idx = {state_id: idx for idx, state_id in enumerate(self.states.keys())}

        for state_id, state in self.states.items():
            from_idx = state_idx[state.id]
            for transition_to_id, multiplier in state.transitions.items():
                to_idx = state_idx[transition_to_id]
                self.transitions[(to_idx, from_idx)] = multiplier
                if isinstance(multiplier, NNOutput):
                    multiplier.add_design_mat_idx((to_idx, from_idx))
                    self.nn_outputs.append(multiplier)


class H(DesignMatrix):
    def __init__(self, states, measures):
        self.states = states
        self.measures = measures
        self.state_to_measures = {}
        super().__init__()

    @property
    def template(self):
        if self._template is None:
            num_measures, num_states = len(self.measures), len(self.states)
            self._template = Variable(torch.zeros((num_measures, num_states)))
            for (row, col), multiplier in self.state_to_measures.items():
                self._template[row, col] = multiplier()
        return self._template

    def register_variables(self):
        self.nn_outputs = []
        self.nn_modules = []

        state_idx = {state_id: idx for idx, state_id in enumerate(self.states.keys())}
        measure_idx = {measure_id: idx for idx, measure_id in enumerate(self.measures.keys())}

        for measure in self.measures.values():
            for state_id, multiplier in measure.states.items():
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
        Take a list of states or measures, and generate a covariance matrix.

        :param variables: A list of states or measures.
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
    def __init__(self, states):
        self.states = states
        super().__init__()

    @property
    def num_vars(self):
        return len(self.states)

    def register_variables(self):
        return self.register_covariance(self.states.values())


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
    # TODO: re-read sections on this to make sure you understand it. then its very similar to H
    pass
