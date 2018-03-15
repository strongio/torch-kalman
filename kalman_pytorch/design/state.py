from kalman_pytorch.design.covariance_element import CovarianceElement
from kalman_pytorch.utils.utils import make_callable
from torch import Tensor


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
                self.transitions[key] = Tensor([self.transitions[key]])
            self.transitions[key] = make_callable(self.transitions[key])


def position_and_velocity(process_std_dev, id_prefix):
    """
    Creates a tuple of States. The first is for position, the second is for velocity. The two are correlated according
    to the "discrete white noise" method (see https://github.com/rlabbe/filterpy/blob/master/filterpy/common/discretization.py).
    This assumes that the separation between each successive timestep is constant.

    :param process_std_dev: Standard deviation (process-noise).
    :param id_prefix: A prefix that will be added to the ids ('position','velocity'). Should include a separator.
    :return: State for Position, State for Velocity.
    """

    # position and velocity:
    position = State(id=id_prefix + 'position', std_dev=process_std_dev.with_added_lambda(lambda x: pow(.5, .5) * x))
    velocity = State(id=id_prefix + 'velocity', std_dev=process_std_dev)
    position.add_correlation(velocity, correlation=1.)

    # next position is just positition + velocity
    position.add_transition(to_state=position)
    velocity.add_transition(to_state=position)
    # next velocity is just current velocity:
    velocity.add_transition(to_state=velocity)

    return position, velocity
