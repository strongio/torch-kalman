from torch_kalman.design.covariance_element import CovarianceElement
from torch_kalman.utils.utils import make_callable
from torch import Tensor

from torch.autograd import Variable


class State(CovarianceElement):
    def __init__(self, id, std_dev):
        """
        A "state" in a kalman filter is an unobserved variable that generates the observed data. At each timestep,
        the state evolves according to transitions and "process-noise".

        :param id: A unique name for this state.
        :param std_dev: The standard-deviation (process-noise).
        """
        super(State, self).__init__(id=id, std_dev=std_dev)
        self.transitions = {}

    def add_transition(self, to_state, multiplier=1.0):
        """
        Specify that this state at T_n links to another state at T_n+1.

        :param to_state: The state that this state links to.
        :param multiplier: The multiplier for the transition (e.g., multipier of .5 means T_n+1 = .5*T_n)
        """
        self.transitions.update({to_state.id: multiplier})

    def torchify(self):
        super(State, self).torchify()
        for key in self.transitions.keys():
            if isinstance(self.transitions[key], float):
                self.transitions[key] = Variable(Tensor([self.transitions[key]]))
            self.transitions[key] = make_callable(self.transitions[key])
