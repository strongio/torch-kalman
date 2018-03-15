from torch.autograd import Variable
from torch import Tensor
from warnings import warn

from kalman_pytorch.utils.utils import make_callable


class KalmanObject(object):
    def __init__(self, id, std_dev):
        """
        Either an measurement or a state.

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
            self.std_dev = Tensor([self.std_dev])
        self.std_dev = make_callable(self.std_dev)

        # correlations:
        for key in self.correlations.keys():
            if isinstance(self.correlations[key], float):
                self.correlations[key] = Tensor([self.correlations[key]])
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


class State(KalmanObject):
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
                self.transitions[key] = Tensor([self.transitions[key]])
            self.transitions[key] = make_callable(self.transitions[key])


class Measurement(KalmanObject):
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
                self.states[key] = Tensor([self.states[key]])
            self.states[key] = make_callable(self.states[key])
