from torch import Tensor
from numpy import nan
from torch.autograd import Variable


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
        if self._design_mat_idx is not None:
            raise Exception("Tried to add the target-index in the design-mat to a NNOutput, but it already has one. This can"
                            " happen if the same NNOutput instance was accidentally re-used on more than one element of the "
                            "design-mat. You can indicate that multiple targets in the design-mat should be filled with the "
                            "same nn-module's output by creating separate NNOutput instances.")
        self._design_mat_idx = idx

    @property
    def design_mat_idx(self):
        if self._design_mat_idx is None:
            raise Exception("Need to `add_design_mat_idx` first.")
        return self._design_mat_idx

    def slice_from_raw(self, raw):
        return raw[:, self.nn_output_idx]


class NNDictOutput(NNOutput):
    def __init__(self, nn_module, nn_output_name):
        super().__init__(nn_module=nn_module, nn_output_idx=nn_output_name)

    def slice_from_raw(self, raw):
        return raw[self.nn_output_idx]
