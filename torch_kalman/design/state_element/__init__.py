from torch_kalman.design.covariance_element import CovarianceElement
from torch_kalman.utils.utils import make_callable
from torch import Tensor

from torch.autograd import Variable

from warnings import warn


class StateElement(CovarianceElement):
    def __init__(self, id, std_dev, initial_mean=0.0, initial_std_dev=None):
        """
        A "state" in a kalman filter is an unobserved variable that generates the observed data. At each timestep,
        the state evolves according to transitions and "process-noise".

        :param id: A unique name for this state.
        :param std_dev: The standard-deviation (process-noise).
        :param initial_mean: The initial value for this state, before any measurements update it.
        :param initial_std_dev: The initial std-deviation for this state, before any measurements update it. Default is to
        use `std_dev`.
        """
        super(StateElement, self).__init__(id=id, std_dev=std_dev)
        self.initial_mean = initial_mean
        if initial_std_dev is None:
            initial_std_dev = std_dev
        self.initial_std_dev = initial_std_dev
        self.transitions = {}

    def add_transition(self, to_state, multiplier=1.0):
        """
        Specify that this state at T_n links to another state at T_n+1.

        :param to_state: The state that this state links to.
        :param multiplier: The multiplier for the transition (e.g., multipier of .5 means T_n+1 = .5*T_n)
        """
        if to_state.id in self.transitions.keys():
            warn("This state ('{}') already has a transition to '{}' recorded. Will overwrite.".format(self.id, to_state.id))
        self.transitions.update({to_state.id: multiplier})

    def torchify(self):
        super(StateElement, self).torchify()
        # initial values:
        if isinstance(self.initial_mean, float):
            self.initial_mean = Variable(Tensor([self.initial_mean]))
        self.initial_mean = make_callable(self.initial_mean)
        if isinstance(self.initial_std_dev, float):
            self.initial_std_dev = Variable(Tensor([self.initial_std_dev]))
        self.initial_std_dev = make_callable(self.initial_std_dev)

        # transitions:
        for key in self.transitions.keys():
            if isinstance(self.transitions[key], float):
                self.transitions[key] = Variable(Tensor([self.transitions[key]]))
            self.transitions[key] = make_callable(self.transitions[key])
