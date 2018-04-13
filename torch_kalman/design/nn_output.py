from collections import defaultdict
from warnings import warn

import torch
from torch import Tensor
from numpy import nan
from torch.autograd import Variable
from torch.nn import ModuleList

from torch_kalman.design.state import NNState


class NNOutput(object):
    def __init__(self, nn_module, nn_output_idx):
        """
        This is a placeholder that indicates a nn-module output will go here.
        """
        self.nan = Tensor([nan])
        self.nn_module = nn_module
        self.nn_output_idx = nn_output_idx
        self._design_mat_idx = None

    def __call__(self, *args, **kwargs):
        return Variable(self.nan)

    def add_design_mat_idx(self, idx):
        self._design_mat_idx = idx

    @property
    def design_mat_idx(self):
        if self._design_mat_idx is None:
            raise Exception("Need to `add_design_mat_idx` first.")
        return self._design_mat_idx

    def pluck_from_raw_output(self, nn_output_raw):
        return nn_output_raw[self.nn_output_idx]


class NNOutputTracker(object):
    def __init__(self):
        self.nn_outputs = []
        self.nn_output_by_module = {}
        self.nn_inputs_by_module = {}
        self._nn_module = None
        self.register_variables()

    def register_variables(self):
        raise NotImplementedError()

    def add_nn_input(self, nn_module, nn_input):
        if self._nn_module is not None:
            raise Exception("Already finalized module.")
        if len(self.nn_outputs) == 0:
            raise Exception("There aren't any NNOutputs.")
        if nn_module in self.nn_inputs_by_module.keys():
            raise Exception("This nn_module has already had its input registered.")
        self.nn_inputs_by_module[nn_module] = nn_input

    def add_nn_inputs(self, modules_and_names):
        for nn_module, nn_input in modules_and_names:
            self.add_nn_input(nn_module, nn_input)

    def finalize_nn_module(self):
        self.nn_output_by_module = defaultdict(list)

        for nn_output in self.nn_outputs:
            if nn_output.nn_module not in self.nn_inputs_by_module.keys():
                raise Exception("One of the nn_modules ('{}'), didn't have its input registered with `add_nn_module_input`.")
            self.nn_output_by_module[nn_output.nn_module].append(nn_output)

        modules = list(self.nn_inputs_by_module.keys())
        self._nn_module = NNModuleConcatenator(modules=modules,
                                               module_inputs=[self.nn_inputs_by_module[module] for module in modules],
                                               module_outputs=[self.nn_output_by_module[module] for module in modules])

    @property
    def nn_module(self):
        if self._nn_module is None:
            raise Exception("The nn_module hasn't been finalized yet with a call to `finalize_nn_module`.")
        return self._nn_module

    @property
    def input_names(self):
        if self._nn_module is None:
            raise Exception("The nn_module hasn't been finalized yet with a call to `finalize_nn_module`.")
        return set(input.name for input in self.nn_module.module_inputs)


class NNModuleConcatenator(torch.nn.Module):
    def __init__(self, modules, module_inputs, module_outputs):
        super().__init__()
        assert len(modules) == len(module_outputs)
        assert len(modules) == len(module_outputs)
        self.modules = ModuleList(modules)
        self.module_inputs = module_inputs
        self.module_outputs = module_outputs

    def isnull(self):
        return len(self.modules) == 0

    def forward(self, time, **kwargs):
        output_list = []
        for i in range(len(self.modules)):
            this_module = self.modules[i]
            this_nn_input = self.module_inputs[i]
            this_input_tens = this_nn_input.slice(kwargs[this_nn_input.name], time=time)
            this_nn_outputs = self.module_outputs[i]
            module_output_raw = this_module(this_input_tens)
            for nn_output in this_nn_outputs:
                output_list.append((nn_output.design_mat_idx, nn_output.pluck_from_raw_output(module_output_raw)))
        return output_list


class DynamicState(NNOutputTracker):
    def __init__(self, states):
        self.states = states
        super().__init__()

    def register_variables(self):
        self.nn_outputs = []

        for idx, (state_id, state) in enumerate(self.states.items()):
            if isinstance(state, NNState):
                state.add_design_mat_idx((idx, 0))
                self.nn_outputs.append(state)

    def update_state_mean(self, state_mean, time, **kwargs):
        if not self.nn_module.isnull:
            nn_module_kwargs = {argname: kwargs[argname][:, time, :] for argname in self.input_names}
            nn_output = self.nn_module(**nn_module_kwargs)
            for (row, col), output in nn_output:
                state_mean[:, row, col] = output
        return state_mean
