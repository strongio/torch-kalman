from warnings import warn

from torch_kalman.design.covariance_element import CovarianceElement
from torch_kalman.utils.utils import make_callable
from torch import Tensor
from torch.autograd import Variable


class Measure(CovarianceElement):
    def __init__(self, id, std_dev):
        """
        A measure in a kalman filter is the observations we actually measure. Measurements are generated from
        their underlying states, plus measure noise.

        :param id: A unique name for this measure.
        :param std_dev: The standard-deviation measure-noise.
        """
        super(Measure, self).__init__(id=id, std_dev=std_dev)
        self.state_elements = {}

    def add_state_element(self, state_element, multiplier=1.0):
        """
        Add a state that generates this measurement.

        :param state_element: A StateElement.
        :param multiplier: The multipler for linking the state to the measure. Typically 1.0.
        """
        if state_element.id in self.state_elements.keys():
            warn("This measure ('{}') is already linked to state_element '{}'. Will overwrite.".
                 format(self.id, state_element.id))
        self.state_elements.update({state_element.id: multiplier})

    def torchify(self):
        super(Measure, self).torchify()
        for key in self.state_elements.keys():
            if isinstance(self.state_elements[key], float):
                self.state_elements[key] = Variable(Tensor([self.state_elements[key]]))
            self.state_elements[key] = make_callable(self.state_elements[key])


# alias that's more consistent w/StateElement:
MeasurementElement = Measure
