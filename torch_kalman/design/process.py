from torch_kalman.design.state import State


def join(iterable, sep):
    """
    :param iterable: An iterable of strings.
    :param sep: A separator.
    :return: `iterable`, with any Nones removed, joined by `sep`.
    """
    return sep.join(el for el in iterable if el is not None)


class Process(object):
    def __init__(self, states):
        self.states = tuple(states)


class NoVelocity(Process):
    def __init__(self, id_prefix, std_dev, sep="_"):
        """
        A no-velocity process. The expected value of the next state is equal to the current one.

        :param id_prefix: A prefix that will be added to the id ('position'). Or can be None for no prefix.
        :param std_dev: Standard deviation (process-noise).
        :param sep: The separator between id_prefix and the state name (defaults "_").
        """
        pos = State(id=join([id_prefix, 'position'], sep), std_dev=std_dev)
        pos.add_transition(pos)
        super(NoVelocity, self).__init__(states=(pos,))

    @property
    def observable(self):
        return self.states[0]


class DampenedVelocity(Process):
    def __init__(self, id_prefix, std_dev, damp_multi, sep="_"):
        """
        A process with velocity, where the expected value of the next state's velocity is equal to, or slightly less
        than, the current one. The position and velocity are correlated according to the "discrete white noise" method
        (see https://github.com/rlabbe/filterpy/blob/master/filterpy/common/discretization.py). This assumes that the
        separation between each successive timestep is constant.

        :param id_prefix: A prefix that will be added to the ids ('position','velocity'). Or can be None.
        :param std_dev: Standard deviation (process-noise).
        :param sep: The separator between id_prefix and the state name (defaults "_").
        :param damp_multi: A multiplier between 0 and 1 that is applied to the velocity at each timestep.
        """
        assert 0 < damp_multi <= 1

        def white_noise_multi(x): return pow(.5, .5) * x

        # position and velocity:
        position = State(id=join([id_prefix, 'position'], sep), std_dev=std_dev.with_added_lambda(white_noise_multi))
        velocity = State(id=join([id_prefix, 'velocity'], sep), std_dev=std_dev)
        position.add_correlation(velocity, correlation=1.)

        # next position is just positition + velocity
        position.add_transition(to_state=position)
        velocity.add_transition(to_state=position)
        # next velocity is just current velocity:
        velocity.add_transition(to_state=velocity, multiplier=damp_multi)

        super(DampenedVelocity, self).__init__((position, velocity))

    @property
    def observable(self):
        return self.states[0]


class ConstantVelocity(DampenedVelocity):
    def __init__(self, id_prefix, std_dev, sep="_"):
        """
        A process with velocity, where the expected value of the next state's velocity is equal to the current one. The
        position and velocity are correltaed according to the "discrete white noise" method
        (see https://github.com/rlabbe/filterpy/blob/master/filterpy/common/discretization.py). This assumes that the
        separation between each successive timestep is constant.

        :param id_prefix: A prefix that will be added to the ids ('position','velocity'). Or can be None.
        :param std_dev: Standard deviation (process-noise).
        :param sep: The separator between id_prefix and the state name (defaults "_").
        """
        super(ConstantVelocity, self).__init__(id_prefix=id_prefix, std_dev=std_dev, damp_multi=1., sep=sep)


class Seasonal(Process):
    def __init__(self, id_prefix, std_dev, period, df_correction=True, sep="_"):
        """
        A seasonal process.

        :param id_prefix: A prefix that will be added to the ids ('season[0-period]'), or can be None.
        :param std_dev: Standard deviation (process-noise).
        :param period: The period for the season (e.g., for daily data, period=7 would be a weekly season).
        :param df_correction: Correct the degrees of freedom? In most contexts this is being used, we also have a process
        that captures the general (non seasonal) level/trend. In that case, we'd want the seasonal process to have
        period - 1 degrees of freedom (df_correction=True, the default).
        :param sep: The separator between parts of the id (defaults "_").
        """
        states = []
        for i in range(period):
            this_state = State(id=join([id_prefix, 'season', str(i)], sep), std_dev=std_dev if i == 0 else 0.0)
            states.append(this_state)

        if df_correction:
            for i in range(period - 1):
                states[i].add_transition(states[i + 1], 1)
                states[i].add_transition(states[0], -1)
        else:
            for i in range(period):
                if i == period - 1:
                    states[i].add_transition(to_state=states[0])
                else:
                    states[i].add_transition(to_state=states[i + 1])

        super(Seasonal, self).__init__(states)

    @property
    def observable(self):
        return self.states[0]
