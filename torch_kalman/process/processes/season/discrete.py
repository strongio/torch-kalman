from math import log
from typing import Optional, Union, Generator, Tuple

import numpy as np
import torch
from numpy.core.multiarray import ndarray
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.utils import fourier_series
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.processes.season.base import DateAware


class Season(DateAware):
    def __init__(self,
                 id: str,
                 seasonal_period: int,
                 season_duration: int = 1,
                 season_start: Optional[str] = None,
                 timestep_interval: Optional[str] = None,
                 use_fourier_init_param: Optional[int] = None):
        """
        Process representing discrete seasons.

        :param id: Unique name for this process
        :param seasonal_period: The number of seasons (e.g. 7 for day_in_week).
        :param season_duration: The length of each season, default 1 time-step.
        :param season_start: A string that can be parsed into a datetime by `numpy.datetime64`. This is when the season
        starts, which is useful to specify if season boundaries are actually meaningful, and is important to specify if
        different groups in your dataset start on different dates.
        :param timestep_interval: A string that is understood as a datetime-unit by numpy.
        See: https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.datetime.html#arrays-dtypes-dateunits
        :param use_fourier_init_param: This determines how the *initial* state-means are parameterized. For longer seasons,
        we may not want a free parameter for each individual season. If an integer, then uses a fourier-series for the
        parameterization, with that integer's (*2) degrees of freedom. If False, then each season (-1) gets a unique starting
         value. If None, then chooses automatically based on season-length.
        """

        self.seasonal_period = seasonal_period
        self.season_duration = season_duration

        # state-elements:
        pad_n = len(str(seasonal_period))
        state_elements = ['measured'] + [str(i).rjust(pad_n, "0") for i in range(1, seasonal_period)]

        # transitions are placeholder, filled in w/batch
        transitions = dict()
        for i in range(seasonal_period):
            current = state_elements[i]
            transitions[current] = {current: None}
            if i > 0:
                prev = state_elements[i - 1]
                transitions[current][prev] = None
                transitions['measured'][prev] = None

        # process-covariance:
        self.log_std_dev = Parameter(torch.randn(1) - 3.)

        # initial state:
        if use_fourier_init_param is None:
            use_fourier_init_param = 4 if self.seasonal_period > 10 else False
        self.K = use_fourier_init_param
        if self.K:
            self.initial_state_mean_params = Parameter(torch.randn((self.K, 2)))
        else:
            self.initial_state_mean_params = Parameter(torch.randn(self.seasonal_period - 1))

        self.initial_state_log_std_dev = Parameter(torch.randn(1) - 3.)

        # super:
        super().__init__(id=id,
                         state_elements=state_elements,
                         transitions=transitions,
                         season_start=season_start,
                         timestep_interval=timestep_interval)

        # writing transition-matrix is slow, no need to do it repeatedly:
        self.transition_cache = {}

    def add_measure(self,
                    measure: str,
                    state_element: str = 'measured',
                    value: Union[float, Tensor, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.log_std_dev
        yield self.initial_state_log_std_dev
        yield self.initial_state_mean_params

    def covariance(self) -> Covariance:
        state_size = len(self.state_elements)
        cov = Covariance(size=(state_size, state_size))
        cov[:] = 0.
        cov[0, 0] = torch.pow(torch.exp(self.log_std_dev), 2)
        return cov

    def for_batch(self,
                  batch_size: int,
                  time: Optional[int] = None,
                  start_datetimes: Optional[ndarray] = None,
                  cache: bool = True
                  ) -> ProcessForBatch:

        if start_datetimes is None:
            if self.start_datetime:
                raise ValueError("`start_datetimes` argument required.")
            delta = np.ones(shape=(batch_size,), dtype=int) * time
        else:
            self.check_datetimes(start_datetimes)
            delta = (start_datetimes - self.start_datetime).view('int64') + time

        in_transition = (delta % self.season_duration) == (self.season_duration - 1)

        for_batch = super().for_batch(batch_size=batch_size)
        if cache:
            key = in_transition.tostring()
            if key not in self.transition_cache.keys():
                self.transition_cache[key] = self.make_batch_transitions(in_transition)
            for_batch.batch_transitions = self.transition_cache[key]
        else:
            for_batch.batch_transitions = self.make_batch_transitions(in_transition)

        return for_batch

    def make_batch_transitions(self, in_transition: np.ndarray) -> ProcessForBatch:
        batch_size = len(in_transition)
        for_batch = super().for_batch(batch_size=batch_size)
        to_next_state = Tensor(in_transition.astype('float'))
        to_self = 1 - to_next_state

        for i in range(1, self.seasonal_period):
            current = self.state_elements[i]
            prev = self.state_elements[i - 1]

            # from state to next state
            for_batch.set_transition(from_element=prev, to_element=current, values=to_next_state)

            # from state to measured:
            if prev == 'measured':  # first requires special-case
                to_measured = Tensor(np.where(in_transition, -1.0, 1.0))
            else:
                to_measured = -to_next_state
            for_batch.set_transition(from_element=prev, to_element='measured', values=to_measured)

            # from state to itself:
            for_batch.set_transition(from_element=current, to_element=current, values=to_self)

        return for_batch.batch_transitions

    def initial_state(self,
                      batch_size: int,
                      start_datetimes: Optional[ndarray] = None,
                      time: Optional[int] = None) -> Tuple[Tensor, Tensor]:

        # mean:
        mean = self.initialize_state_mean()

        if start_datetimes is None:
            if self.start_datetime:
                raise ValueError("`start_datetimes` argument required.")
            offset = np.zeros(shape=(batch_size,), dtype=int)
        else:
            self.check_datetimes(start_datetimes)
            offset = (start_datetimes - self.start_datetime).view('int64')

        season_shift = np.floor(offset / self.season_duration) % self.seasonal_period

        means = []
        for i, shift in enumerate(season_shift.astype(int)):
            if i == 0:
                means.append(mean)
            else:
                means.append(torch.cat([mean[-shift:], mean[:-shift]]))
        means = torch.stack(means)

        # cov:
        covs = torch.eye(len(self.state_elements)).expand(batch_size, -1, -1)
        covs[:, 0, 0] = torch.pow(torch.exp(self.initial_state_log_std_dev), 2)

        return means, covs

    def initialize_state_mean(self):
        if self.K:
            # fourier series:
            season = torch.arange(float(self.seasonal_period))
            state = fourier_series(time=season,
                                   seasonal_period=self.seasonal_period,
                                   parameters=self.initial_state_mean_params)
            # adjust for df:
            state = state - state[1:].mean()
            state[1:] = state[1:] - state[0] / self.seasonal_period
            state[0] = -torch.sum(state[1:])
        else:
            state = Tensor(size=(self.seasonal_period,))
            state[1:] = self.initial_state_mean_params
            state[0] = -torch.sum(state[1:])
        return state

    def set_to_simulation_mode(self):
        super().set_to_simulation_mode()

        self.initial_state_mean_params[:] = 0.
        self.initial_state_mean_params[:, 0] = log(len(self.state_elements)) / 2.
        self.initial_state_log_std_dev[:] = -10.0  # season-effect virtually identical for all groups
        self.log_std_dev[:] = -8.0  # season-effect is very close to stationary
