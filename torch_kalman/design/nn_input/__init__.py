"""
When designing a kalman filter, we might want some aspects of the design to be dynamic, rather than fixed. We can do this by
passing a torch.nn.module to the Design, using the `add_nn_module`. Designs can have multiple nn-modules, e.g., one that
determines the initial state-mean, one that determines the transition matrix F, etc.

In order to specify which `forward` kwargs should be passed to which nn-modules, we use NNInputs. These allow us to specify
the kwarg names for the nn-module. For inputs that are tensors, we also want to slice the tensor for the given timepoint.
"""


class NNInput(object):
    def __init__(self, name):
        self.name = name


class NNTensorInput(NNInput):

    def slice(self, tensor, time):
        raise NotImplementedError()


class CurrentTime(NNTensorInput):
    """
    This input is for slicing only the current timepoint (from a tensor which has all timepoints).
    """

    def __init__(self, name, num_dims=3):
        super().__init__(name=name)
        assert num_dims == 2 or num_dims == 3
        self.num_dims = num_dims

    def slice(self, tensor, time):
        if self.num_dims == 3:
            return tensor[:, time, :]
        else:
            return tensor[:, time]


class UpToCurrentTime(NNTensorInput):
    """
    This input is for slicing all times up to (but not including) the current time (from a tensor which has all timepoints).
    """

    def __init__(self, name, num_dims=3):
        super().__init__(name=name)
        assert num_dims == 2 or num_dims == 3
        self.num_dims = num_dims

    def slice(self, tensor, time):
        if self.num_dims == 3:
            return tensor[:, 0:time, :]
        else:
            return tensor[:, 0:time]


class InitialToCurrentTime(NNTensorInput):
    """
    This input is for specifying the initial time (in absolute units), which is then used to calculate the current time.
    """

    def __init__(self, name):
        super().__init__(name=name)

    def slice(self, tensor, time):
        if len(tensor.data.shape) != 1:
            raise ValueError("The input {} should be one-dimensional.")
        return tensor + time
