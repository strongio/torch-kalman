import torch
from torch.autograd import Variable

from torch_kalman.design.design_matrix import DesignMatrix
from torch_kalman.design.nn_output import NNOutput


class InitialState(DesignMatrix):
    def __init__(self, states):
        """
        Not really a design-matrix, but uses almost all the same methods.

        :param states: The States.
        """
        self.states = states
        self.initial_state_means = {}
        self.initial_state_std_devs = {}
        super().__init__()
        self._means_template = None
        self._cov_template = None

    @property
    def template(self):
        raise NotImplementedError("`InitialState` has two types of templates, refer to `means_template` or `cov_template`.")

    @property
    def means_template(self):
        if self._means_template is None:
            num_states = len(self.states)
            self._means_template = Variable(torch.zeros(num_states, 1))
            for i, initial_mean in self.initial_state_means.items():
                self._means_template[i] = initial_mean()
        return self._means_template

    @property
    def cov_template(self):
        if self._cov_template is None:
            num_states = len(self.states)
            self._cov_template = Variable(torch.zeros(num_states, num_states))
            for i, initial_std_dev in self.initial_state_std_devs.items():
                self._cov_template[i, i] = torch.pow(initial_std_dev(), 2)
        return self._cov_template

    def register_variables(self):
        self.nn_outputs = []

        for i, (state_id, state) in enumerate(self.states.items()):
            # state mean:
            self.initial_state_means.update({i: state.initial_mean})
            if isinstance(state.initial_mean, NNOutput):
                state.initial_mean.add_design_mat_idx((i, 0))
                self.nn_outputs.append(state.initial_mean)

            # state std-dev:
            self.initial_state_std_devs.update({i: state.initial_std_dev})
            if isinstance(state.initial_std_dev, NNOutput):
                raise NotImplementedError("Found a NNOutput initial state std-dev, which is not currently supported.")

    def create_means_for_batch(self, time, **kwargs):
        if time != 0:
            raise Exception("InitialState is for time-zero only.")

        return super()._create_for_batch(time=time, template=self.means_template, **kwargs)

    def create_cov_for_batch(self, time, **kwargs):
        if time != 0:
            raise Exception("InitialState is for time-zero only.")

        return super()._create_for_batch(time=time, template=self.cov_template, **kwargs)
