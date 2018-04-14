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

        super().__init__(horizon=horizon, design=Design())
        self.measures = [str(x) for x in measures]
        if 'common' in self.measures:
            raise ValueError("'common' is a reserved name, and can't be a measure-name.")
        self._vel_dampening = None
        self.init_params = ParameterList()
        self.process_params = ParameterList()
        self.measure_std_params = ParameterList()
        self.measure_corr_params = ParameterList()
        self.processes_per_dim = {measure_name: list() for measure_name in self.measures_and_common}

    def add_process(self, measure_name, process):
        # add process to design:
        self.design.add_states(process.states)

        # keep track of the measure it belongs to:
        measure_processes = self.processes_per_dim[measure_name]

        process_name = process.__class__.__name__
        if process_name != 'Seasonal':
            if process_name in set(x.__class__.__name__ for x in measure_processes):
                warn("Already added process '{}' to measure '{}'.".format(process_name, measure_name))

        measure_processes.append(process)

    def add_level(self, measures):
        for measure_name in measures:
            self.process_params.append(Param0())
            self.init_params.append(Param0())

            process = NoVelocity(id_prefix=measure_name,
                                 std_dev=LogLinked(self.process_params[-1]),
                                 initial_value=self.init_params[-1])

            self.add_process(measure_name, process)

    def add_trend(self, measures):
        for measure_name in measures:
            self.process_params.append(Param0(2))
            self.init_params.append(Param0())

            process = DampenedVelocity(id_prefix=measure_name,
                                       std_devs=LogLinked(self.process_params[-1]),
                                       initial_position=self.init_params[-1],
                                       damp_multi=LogitLinked(self.vel_dampening))

            self.add_process(measure_name, process)

    def add_season(self, measures, period, duration):
        for measure_name in measures:
            self.process_params.append(Param0())

            process = Seasonal(id_prefix=measure_name,
                               period=period,
                               std_dev=LogLinked(self.process_params[-1]),
                               duration=duration,
                               time_input_name='timestep_abs')

            self.add_process(measure_name, process)
            process.add_modules_to_design(self.design, known_to_super=False)

    def finalize(self):
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
                    this_measure.add_state(process.observable)

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

    def parameters(self):
        if not self.design.finalized:
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
