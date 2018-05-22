from torch_kalman.lazy_parameter import LazyParameter
from torch_kalman.design.process import Process
from torch_kalman.utils.utils import nonejoin
from torch_kalman.design.state_element import StateElement


class NoVelocity(Process):
    def __init__(self, id_prefix, std_dev, initial_value, sep="_"):
        """
        A no-velocity process. The expected value of the next state is equal to the current one.

        :param id_prefix: A prefix that will be added to the id ('position'). Or can be None for no prefix.
        :param std_dev: Standard deviation (process-noise).
        :param sep: The separator between id_prefix and the state name (defaults "_").
        """
        pos = StateElement(id=nonejoin([id_prefix, 'position'], sep), std_dev=std_dev, initial_mean=initial_value)
        pos.add_transition(pos)
        super(NoVelocity, self).__init__(states=(pos,))

    @property
    def observable(self):
        return self.states[0]


class DampenedVelocity(Process):
    def __init__(self, id_prefix, std_devs, corr=0.0, initial_position=0.0, damp_multi=1.0, sep="_"):
        """
        A process with velocity, where the expected value of the next state's velocity is equal to, or slightly less
        than, the current one.

        :param id_prefix:
        :param std_devs:
        :param corr:
        :param initial_position:
        :param damp_multi:
        :param sep:
        """

        # position and velocity:
        position = StateElement(id=nonejoin([id_prefix, 'position'], sep), std_dev=std_devs[0], initial_mean=initial_position)
        velocity = StateElement(id=nonejoin([id_prefix, 'velocity'], sep), std_dev=std_devs[1])
        position.add_correlation(velocity, correlation=corr)

        # next position is just positition + velocity
        position.add_transition(to_state_element=position)
        velocity.add_transition(to_state_element=position)
        # next velocity is just current velocity:
        velocity.add_transition(to_state_element=velocity, multiplier=damp_multi)

        super(DampenedVelocity, self).__init__((position, velocity))

    @property
    def observable(self):
        return self.states[0]


class ConstantVelocity(DampenedVelocity):
    def __init__(self, id_prefix, std_devs, corr, initial_position=0.0, sep="_"):
        """
        A process with velocity, where the expected value of the next state's velocity is equal to the current one.

        :param id_prefix:
        :param std_devs:
        :param corr:
        :param initial_position:
        :param sep:
        """
        super().__init__(id_prefix=id_prefix,
                         std_devs=std_devs, corr=corr, damp_multi=1., initial_position=initial_position, sep=sep)


class PhysicsVelocity(DampenedVelocity):
    white_noise_multi = lambda x: pow(.5, .5) * x

    def __init__(self, id_prefix, std_dev, initial_position, damp_multi=1.0, sep="_"):
        """
        A DampenedVelocity process where the position and velocity are correlated according to the "discrete white noise"
        method (see https://github.com/rlabbe/filterpy/blob/master/filterpy/common/discretization.py). This assumes that the
        separation between each successive timestep is constant.

        :param id_prefix:
        :param std_dev:
        :param initial_position:
        :param damp_multi:
        :param sep:
        """
        if not isinstance(std_dev, LazyParameter):
            std_dev = LazyParameter(std_dev)

        std_devs = [std_dev.with_added_lambda(self.white_noise_multi), std_dev]

        super().__init__(id_prefix=id_prefix, std_devs=std_devs, corr=1.0,
                         initial_position=initial_position, damp_multi=damp_multi, sep=sep)