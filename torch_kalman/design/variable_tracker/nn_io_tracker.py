from warnings import warn

from collections.__init__ import defaultdict

from torch_kalman.design.nn_module_concatenator import NNModuleConcatenator
from torch_kalman.design.variable_tracker import VariableTracker


class NNIOTracker(VariableTracker):
    def __init__(self):
        self.nn_outputs = []
        self._nn_module = None
        super().__init__()

    def register_variables(self):
        raise NotImplementedError()

    @property
    def nn_module(self):
        if self._nn_module is None:
            raise Exception("The nn_module hasn't been finalized yet with a call to `prepare_nn_module`.")
        return self._nn_module

    @property
    def input_names(self):
        if self._nn_module is None:
            raise Exception("The nn_module hasn't been finalized yet with a call to `prepare_nn_module`.")
        return set(input.name for input in self.nn_module.module_inputs)

    def prepare_nn_module(self, nn_inputs_by_module):
        """
        :param nn_inputs_by_module: A dictionary where torch.nn.Module instances are the keys, and lists of NNInputs are the
        values.
        """
        assert self._nn_module is None, "The nn_module has already been finalized."

        nn_outputs_by_module = defaultdict(list)
        for nn_output in self.nn_outputs:
            if nn_output.nn_module not in nn_inputs_by_module.keys():
                raise Exception("One of the nn_outputs' `nn_module` was not found in `submodules_and_inputs.keys()`:\n{}".
                                format(nn_output.nn_module))
            nn_outputs_by_module[nn_output.nn_module].append(nn_output)

        self._nn_module = NNModuleConcatenator()

        for nn_module, nn_outputs in nn_outputs_by_module.items():
            nn_inputs = nn_inputs_by_module.get(nn_module, None)
            if nn_inputs is None:
                warn("The following nn_module has an output, but no inputs:\n{}".format(nn_module))
                nn_inputs = []

            self._nn_module.add_module_io(nn_module=nn_module, nn_inputs=nn_inputs, nn_outputs=nn_outputs)

        self._nn_module.finalize()