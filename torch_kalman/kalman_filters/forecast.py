from torch_kalman.design import Design
from torch_kalman.design.process.seasonal import Seasonal
from torch_kalman.design.process.velocity import NoVelocity, DampenedVelocity
from torch_kalman.design.measure import Measure
from torch_kalman.kalman_filter import KalmanFilter

from torch_kalman.design.lazy_parameter import LogLinked, LogitLinked

from torch.nn import ParameterList

from warnings import warn

from torch_kalman.utils.torch_utils import Param0


class Forecast(KalmanFilter):
    def __init__(self, measures, horizon):
        """
        :param measures: The measurable dimensions that will be forecasted.
        :param horizon: The forecast horizon used by "forward".
        """

        super().__init__(horizon=horizon, design=Design())
        self.measures = [str(x) for x in measures]
        if 'common' in self.measures:
            raise ValueError("'common' is a reserved name, and can't be a measure-name.")
        self._vel_dampening = None
        self.init_params = ParameterList()
        self.process_params = ParameterList()
        self.measure_std_params = ParameterList()
        self.measure_corr_params = ParameterList()
        self.state_to_measure_params = ParameterList()
        self.processes_per_dim = {measure_name: list() for measure_name in self.measures_and_common}

    def add_level(self, measure_name):
        """
        Add a "no-velocity" process to `measure_name`. This is a process that assumes that the underlying state that
        generates the measurement does not have any velocity, so its expected value at the next timepoint is the same as its
        current value.

        :param measure_name: The name of the measure.
        """
        assert isinstance(measure_name, str)

        self.process_params.append(Param0())
        self.init_params.append(Param0())

        process = NoVelocity(id_prefix=measure_name,
                             std_dev=LogLinked(self.process_params[-1]),
                             initial_value=self.init_params[-1])

        self.add_process(measure_name, process)

    def add_trend(self, measure_name):
        """
        Add a "constant-velocity" process to `measure_name`. This is a process that assumes that the underlying state that
        generates the measurement has a velocity. As measurements accumulate, the latent position and velocity will be jointly
        estimated, and predictions will be made by combining them.

        :param measure_name: The name of the measure.
        """
        assert isinstance(measure_name, str)

        self.process_params.append(Param0(2))
        self.init_params.append(Param0())

        process = DampenedVelocity(id_prefix=measure_name,
                                   std_devs=LogLinked(self.process_params[-1]),
                                   initial_position=self.init_params[-1],
                                   damp_multi=LogitLinked(self.vel_dampening))

        self.add_process(measure_name, process)

    def add_season(self, measure_name, period, duration, season_start=None, time_start_input_name='time_start'):
        """
        Add a seasonal process to `measure_name`.

        :param measure_name: The name of the measure.
        :param period: The period of the seasonality (i.e., how many seasons pass before we return to the first season).
        :param duration: The duration of the seasonality (i.e., how many timesteps pass before the season changes). For
        example, if we wanted to indicate week-in-year seasonality for daily data, we'd specify period=52, duration=7.
        :param season_start: The timestep on which the season-starts. This value is in the same units as those given by
        the next argument.
        :param time_start_input_name: When `forward` is called, you need to provide an argument with this name, specifying
        the timestep at which each group starts.
        """
        assert isinstance(measure_name, str)
        self.process_params.append(Param0())

        process = Seasonal(id_prefix=measure_name,
                           period=period,
                           std_dev=LogLinked(self.process_params[-1]),
                           duration=duration,
                           season_start=season_start,
                           time_start_input_name=time_start_input_name)

        self.add_process(measure_name, process)

        process.add_modules_to_design(self.design, known_to_super=False)

    def add_common(self, type, measures=None, **kwargs):
        """
        Add a process that is common to several or all measures.

        Later Forecast will be set-up to accept arbitrary joining of measures into "common" levels. That functionality isn't
        present yet, but didn't want to have to make a backwards-incompatible API change.

        :param type: "Level", "trend", or "season".
        :param measures: The measures that will have a common state underlying them.
        :param kwargs: Kwargs passed to the add_* method.
        """
        measures = measures or self.measures  # default
        if set(measures) != set(self.measures):
            raise ValueError("Currently `common` only supported if it applies to all measures.")

        if type == 'level':
            self.add_level(measure_name='common')
        elif type == 'trend':
            self.add_trend(measure_name='common')
        elif type == 'season':
            self.add_season(measure_name='common', **kwargs)
        else:
            raise ValueError("Unrecognized type.")

    def add_process(self, measure_name, process):
        """
        Helper for the add_* methods.

        :param measure_name: The measure-name
        :param process: The Process class.
        """
        if self.finalized:
            raise Exception("Cannot add processes to finalized design.")

        # add process to design:
        self.design.add_state_elements(process.states)

        # keep track of the measure it belongs to:
        measure_processes = self.processes_per_dim[measure_name]

        process_name = process.__class__.__name__
        if process_name != 'Seasonal':
            if process_name in set(x.__class__.__name__ for x in measure_processes):
                warn("Already added process '{}' to measure '{}'.".format(process_name, measure_name))

        measure_processes.append(process)

    def finalize(self):
        """
        Finalize this Forecast so it can be used. Will be called automatically by `nn.module().parameters` (so typically when
         the optimizer is created).
        """
        if self.finalized:
            raise Exception("Already finalized.")

        if sum(len(x) for x in self.processes_per_dim.values()) == 0:
            raise ValueError("Need to add at least one process (level/trend/season).")

        for i, measure_name in enumerate(self.measures):

            # create measure:
            self.measure_std_params.append(Param0())
            this_measure = Measure(id=measure_name,
                                   std_dev=LogLinked(self.measure_std_params[-1]))

            # specify the states that go into this measure:
            for name in (measure_name, 'common'):
                for process in self.processes_per_dim.get(name, []):
                    multiplier = 1.0
                    this_measure.add_state_element(process.observable, multiplier=multiplier)

            # add to design:
            self.design.add_measure(this_measure)

        # correlation between measure-errors (currently constrained to be positive)
        for row in range(self.num_measures):
            for col in range(row + 1, self.num_measures):
                m1 = self.design.measures[self.measures[row]]
                m2 = self.design.measures[self.measures[col]]
                self.measure_corr_params.append(Param0())
                m1.add_correlation(m2, correlation=LogitLinked(self.measure_corr_params[-1]))

        # finalize design:
        self.design.finalize()

    @property
    def finalized(self):
        return self.design.finalized

    def parameters(self):
        """
        Make sure the Forecast is finalized before it's used.
        :return: A generator of parameters.
        """
        if not self.finalized:
            self.finalize()
        return super().parameters()

    @property
    def measures_and_common(self):
        return self.measures + ['common']

    @property
    def vel_dampening(self):
        if self._vel_dampening is None:
            self._vel_dampening = Param0()
        return self._vel_dampening

    @property
    def num_measures(self):
        return len(self.measures)

    @property
    def num_correlations(self):
        return int(((len(self.measures) + 1) * len(self.measures)) / 2 - len(self.measures))
