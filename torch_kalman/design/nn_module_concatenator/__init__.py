import torch
from torch.nn import ModuleList


class NNModuleConcatenator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.concat_modules = ModuleList()
        self.input_lists = []
        self.output_lists = []
        self.finalized = False

    def add_module_io(self, nn_module, nn_inputs, nn_outputs):
        self.concat_modules.append(nn_module)
        self.input_lists.append(nn_inputs)
        self.output_lists.append(nn_outputs)

    def finalize(self):
        self.finalized = True

    def parameters(self):
        assert self.finalized
        super().parameters()

    @property
    def isnull(self):
        return len(self.output_lists) == 0

    def forward(self, time, **kwargs):
        assert self.finalized
        for i, this_module in enumerate(self.concat_modules):
            this_kwargs = {this_nn_input.name: this_nn_input.slice(kwargs[this_nn_input])
                           for this_nn_input in self.input_lists[i]}
            this_nn_outputs = self.output_lists[i]
            module_output_raw = this_module(**this_kwargs)
            for nn_output in this_nn_outputs:
                yield (nn_output.design_mat_idx, nn_output.slice_from_raw(module_output_raw))