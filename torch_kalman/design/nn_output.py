from torch import Tensor
from numpy import nan
from torch.autograd import Variable


class NNOutput(object):
    def __init__(self):
        self.nan = Tensor([nan])

    def __call__(self, *args, **kwargs):
        return Variable(self.nan)
