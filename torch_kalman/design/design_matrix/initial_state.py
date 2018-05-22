import torch
from torch.autograd import Variable

from torch_kalman.design.design_matrix import DesignMatrix
from torch_kalman.design.nn_output import NNOutput
from torch_kalman.utils.torch_utils import expand


class InitialState(DesignMatrix):
    def __init__(self, state_elements):
        """
        Not really a design-matrix, but uses almost all the same methods.

        :param state_elements: The StatesElements.
        """
        self.state_elements = state_elements
        self.initial_state_means = {}
        self.initial_state_std_devs = {}
        super().__init__()
        self._means_template = None
        self._cov_template = None
        self._template = {'means': self.means_template, 'cov': self.cov_template}

    def reset(self):
        self._means_template = None
        self._cov_template = None
        self.batch_cache = {}

    @property
    def template(self):
        raise NotImplementedError("`InitialState` has two types of templates, refer to `means_template` or `cov_template`.")

    @property
    def means_template(self):
        if self._means_template is None:
            num_state_elements = len(self.state_elements)
            self._means_template = Variable(torch.zeros(num_state_elements, 1))
            for i, initial_mean in self.initial_state_means.items():
                self._means_template[i] = initial_mean()
        return self._means_template

    @property
    def cov_template(self):
        if self._cov_template is None:
            num_state_elements = len(self.state_elements)
            self._cov_template = Variable(torch.zeros(num_state_elements, num_state_elements))
            for i, initial_std_dev in self.initial_state_std_devs.items():
                self._cov_template[i, i] = torch.pow(initial_std_dev(), 2)
        return self._cov_template

    def register_variables(self):
        self.nn_outputs = []

        for i, (_, state_element) in enumerate(self.state_elements.items()):
            # state_element mean:
            self.initial_state_means.update({i: state_element.initial_mean})
            if isinstance(state_element.initial_mean, NNOutput):
                state_element.initial_mean.add_design_mat_idx((i, 0))
                self.nn_outputs.append(state_element.initial_mean)

            # state_element std-dev:
            self.initial_state_std_devs.update({i: state_element.initial_std_dev})
            if isinstance(state_element.initial_std_dev, NNOutput):
                raise NotImplementedError("Found a NNOutput for a state's std-dev, which is not currently supported.")

    def create_means_for_batch(self, time, **kwargs):
        if time != 0:
            raise Exception("InitialState is for time-zero only.")

        return super()._create_for_batch(time=time, template=self.means_template, **kwargs)

    def create_cov_for_batch(self, time, **kwargs):
        if time != 0:
            raise Exception("InitialState is for time-zero only.")
        bs = kwargs['kf_input'].data.shape[0]

        # no NNOutput supported, so just expand
        return expand(self.cov_template, bs)
