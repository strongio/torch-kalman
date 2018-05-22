from torch_kalman.design.state_element import StateElement


class NNStateElement(StateElement):
    def __init__(self, id, nn_module, nn_output_idx, std_dev, initial_mean=0.0, initial_std_dev=None):
        """
        A NN-state is a state whose value is determined, not by the kalman-filter algorithm, but by an external callable
        that's called on each timestep. This callable (usually a nn.module) takes the batch as input and returns a Variable,
        the elements of which fill the NNStates.

        :param id:
        :param nn_module:
        :param nn_output_idx:
        :param std_dev:
        :param initial_mean:
        :param initial_std_dev:
        """
        self.nn_module = nn_module
        self.nn_output_idx = nn_output_idx
        self._design_mat_idx = None
        super().__init__(id=id, std_dev=std_dev, initial_mean=initial_mean, initial_std_dev=initial_std_dev)

    def add_transition(self, to_state, multiplier=1.0):
        raise NotImplementedError("NNStates cannot have transitions.")

    def add_correlation(self, obj, correlation):
        raise NotImplementedError("NNStates cannot have process-covariance.")

    def add_design_mat_idx(self, idx):
        self._design_mat_idx = idx

    @property
    def design_mat_idx(self):
        if self._design_mat_idx is None:
            raise Exception("Need to `add_design_mat_idx` first.")
        return self._design_mat_idx

    def pluck_from_raw_output(self, nn_output_raw):
        return nn_output_raw[:, self.nn_output_idx]