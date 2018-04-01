from torch import Tensor
from torch.autograd import Variable

from warnings import warn

from torch_kalman.utils.utils import make_callable


class CovarianceElement(object):
    def __init__(self, id, std_dev):
        """
        The element of a covariance matrix. This is probably either an measurement or a state.

        :param id: A unique ID/name for this state/measurement.
        :param std_dev: The standard-deviation (process-noise/measurement-noise). This is typically a pytorch
        Parameter.
        """
        self.id = id
        self.std_dev = std_dev
        self.correlations = {}

    def torchify(self):
        """
        This function allows arguments like `std_dev` and `correlation` to be more flexible (taking floats, Variables,
        or functions applied to variables). It converts floats to Tensors, and converts non-callable objects like
        floats, Tensors, Parameters, etc. to callables. The latter allows the user to use link functions on their
        Parameters.
        """
        # std-dev:
        if isinstance(self.std_dev, float):
            self.std_dev = Variable(Tensor([self.std_dev]))
        self.std_dev = make_callable(self.std_dev)

        # correlations:
        for key in self.correlations.keys():
            if isinstance(self.correlations[key], float):
                self.correlations[key] = Variable(Tensor([self.correlations[key]]))
            self.correlations[key] = make_callable(self.correlations[key])

    @staticmethod
    def equals(x, y):
        """
        A check for equality that works on both floats, Tensors, and Variables.

        :param x: A float or Tensor of Variable
        :param y: A float or Tensor of Variable
        :return: True or False.
        """
        if isinstance(x, (Tensor, Variable)) or isinstance(y, (Tensor, Variable)):
            return (x == y).numpy()
        else:
            return x == y

    def add_correlation(self, obj, correlation):
        """
        Add a correlation between this state/measurement and another. This will modify both self and the passed obj.

        :param obj: A state or measurement.
        :param correlation: The correlation coefficient. Can be a float, Tensor, Variable, or Parameter.
        """
        # check that classes match:
        if not isinstance(obj, self.__class__):
            raise Exception("Class mismatch (%s,%s)." % (self.__class__, obj.__class__))

        # modify own correlation:
        self.correlations.update({obj.id: correlation})

        # check for match or add to obj:
        other_corr = obj.correlations.get(self.id)
        if other_corr is not None:
            if self.equals(other_corr, correlation):
                return
            else:
                warn("Correlation between these two already present, will overwrite.")
        obj.add_correlation(self, correlation)
