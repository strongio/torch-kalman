from torch_kalman.design.nn_output import NNOutput, NNDictOutput
from torch_kalman.design.process import Process
from torch_kalman.design.process.seasonal.nn_modules import InitialSeasonStateNN, SeasonDurationNN, SeasonNNInput
from torch_kalman.utils.utils import nonejoin
from torch_kalman.design.state import State


class Seasonal(Process):
    def __init__(self, id_prefix, std_dev, period, duration, season_start=None, time_start_input_name='time_start', sep="_"):
        """

        :param id_prefix: A string that will be prepended to the name of the states that make up this process.
        :param std_dev: The std-deviation of the seasonal process.
        :param period: The period of the seasonality (i.e., how many seasons pass before we return to the first season).
        :param duration: The duration of the seasonality (i.e., how many timesteps pass before the season changes). For
        example, if we wanted to indicate week-in-year seasonality for daily data, we'd specify period=52, duration=7.
        :param season_start: The timestep on which the season-starts. This value is in the same units as those given by
        the next argument.
        :param time_start_input_name: When `forward` is called, you need to provide an argument with this name, specifying
        the timestep at which each group starts.
        :param sep: Separator for creating state ids from prefix and individual ids.
        """
        if (duration == 1) != (not season_start):
            if duration == 1:
                raise ValueError("If duration == 1, then do not supply `season_start`.")
            else:
                raise ValueError("If duration > 1, must supply `season_start")

        # input:
        self.nn_input = SeasonNNInput(name=time_start_input_name)

        # define states, their std-dev, and initial-values:
        self.nn_module_initial = InitialSeasonStateNN(period=period, duration=duration)
        pad_n = len(str(period))
        states = []
        for i in range(period):
            season = str(i).rjust(pad_n, "0")
            this_state = State(id=nonejoin([id_prefix, 'season', season], sep),
                               std_dev=std_dev if i == 0 else 0.0,
                               initial_value=NNOutput(nn_module=self.nn_module_initial, nn_output_idx=i))
            states.append(this_state)

        # define transitions:
        if duration > 1:
            self.nn_module_season_duration = SeasonDurationNN(period=period, duration=duration, season_start=season_start)
            multipliers = {'to_next': lambda: NNDictOutput(self.nn_module_season_duration, nn_output_name='to_next'),
                           'from_first_to_first': lambda: NNDictOutput(self.nn_module_season_duration,
                                                                       nn_output_name='from_first_to_first'),
                           'to_first': lambda: NNDictOutput(self.nn_module_season_duration, nn_output_name='to_first'),
                           'to_self': lambda: NNDictOutput(self.nn_module_season_duration, nn_output_name='to_self')}
        else:
            self.nn_module_season_duration = None
            multipliers = {'to_next': lambda: 1.,
                           'to_first': lambda: -1.,
                           'from_first_to_first': lambda: -1.,
                           'to_self': lambda: 0.}

        # for the first state, only need to define two transitions:
        states[0].add_transition(to_state=states[1], multiplier=multipliers['to_next']())
        states[0].add_transition(to_state=states[0], multiplier=multipliers['from_first_to_first']())

        # for the rest, need to define three transitions:
        for i in range(1, period):
            if (i + 1) < period:
                # when transitioning:
                states[i].add_transition(to_state=states[i + 1], multiplier=multipliers['to_next']())
                states[i].add_transition(to_state=states[0], multiplier=multipliers['to_first']())

            # when not transitioning:
            states[i].add_transition(to_state=states[i], multiplier=multipliers['to_self']())

        super(Seasonal, self).__init__(states)

    def add_modules_to_design(self, design, known_to_super=False):
        design.add_nn_module(type='initial_state',
                             nn_module=self.nn_module_initial,
                             nn_input=self.nn_input,
                             known_to_super=known_to_super)
        if self.nn_module_season_duration is not None:
            design.add_nn_module(type='transition',
                                 nn_module=self.nn_module_season_duration,
                                 nn_input=self.nn_input,
                                 known_to_super=known_to_super)

    @property
    def observable(self):
        return self.states[0]
