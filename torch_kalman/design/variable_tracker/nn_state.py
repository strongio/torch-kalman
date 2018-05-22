from torch_kalman.design.state_element.nn_state_element import NNStateElement
from torch_kalman.design.variable_tracker.nn_io_tracker import NNIOTracker


class NNState(NNIOTracker):
    def __init__(self, state_elements):
        self.state_elements = state_elements
        super().__init__()

    def register_variables(self):
        self.nn_outputs = []

        for idx, (state_id, state_element) in enumerate(self.state_elements.items()):
            if isinstance(state_element, NNStateElement):
                state_element.add_design_mat_idx((idx, 0))
                self.nn_outputs.append(state_element)

    def update_state_mean(self, state_mean, time, **kwargs):
        if not self.nn_module.isnull:
            nn_module_kwargs = {argname: kwargs[argname][:, time, :] for argname in self.input_names}
            nn_output = self.nn_module(**nn_module_kwargs)
            for (row, col), output in nn_output:
                state_mean[:, row, col] = output
