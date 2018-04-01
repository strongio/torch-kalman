from torch_kalman.design.covariance_element import CovarianceElement
from torch_kalman.utils.utils import make_callable
from torch import Tensor
from torch.autograd import Variable


class Measurement(CovarianceElement):
    def __init__(self, id, std_dev):
        """
        A measurement in a kalman filter is the observations we actually measure. Measurements are generated from
        their underlying states, plus measurement noise.

        :param id: A unique name for this measurement.
        :param std_dev: The standard-deviation measurement-noise.
        """
        super(Measurement, self).__init__(id=id, std_dev=std_dev)
        self.states = {}

    def add_state(self, state, multiplier=1.0):
        """
        Add a state that generates this measurement.

        :param state: A State.
        :param multiplier: The multipler for linking the state to the measurement. Typically 1.0.
        """
        self.states.update({state.id: multiplier})

    def torchify(self):
        super(Measurement, self).torchify()
        for key in self.states.keys():
            if isinstance(self.states[key], float):
                self.states[key] = Variable(Tensor([self.states[key]]))
            self.states[key] = make_callable(self.states[key])
