import torch

from torch_kalman.lazy_parameter import LazyParameter


class LogLinked(LazyParameter):
    def __init__(self, parameter):
        super(LogLinked, self).__init__(parameter=parameter)
        self.lambda_chain.append(torch.exp)
        self.after_init_idx = 1


class LogitLinked(LazyParameter):
    def __init__(self, parameter):
        super(LogitLinked, self).__init__(parameter=parameter)
        self.lambda_chain.append(torch.sigmoid)
        self.after_init_idx = 1