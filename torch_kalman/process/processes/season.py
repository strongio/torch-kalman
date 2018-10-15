from math import log
from typing import Generator, Union, Tuple, Optional
from warnings import warn

import torch

import numpy as np

from numpy.core.multiarray import ndarray
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch


class Season(Process):
    def __init__(self,
                 id: str,
                 num_seasons: int,
                 season_duration: int = 1,
                 season_start: Optional[str] = None,
                 timestep_interval: Optional[str] = None,
                 use_fourier_init_param: Optional[int] = None):
        """

        :param id: Unique name for this process
        :param num_seasons: The number of seasons (e.g. 7 for day_in_week).
        :param season_duration: The length of each season, default 1 time-step.
        :param season_start: A string that can be parsed into a datetime by `numpy.datetime64`. This is when the season
        starts, which is useful to specify if season boundaries are actually meaningful, and is important to specify if
        different groups in your dataset start on different dates.
        :param timestep_interval: A string that is understood as a datetime-unit by numpy.
        See: https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.datetime.html#arrays-dtypes-dateunits
        :param use_fourier_init_param: This determines how the initial seasons are parameterized. For longer seasons, we may
         not want a free parameter for each individual season. If an integer, then uses a fourier-series for the
         parameterization, with that integer's (*2) degrees of freedom. If False, then each season (-1) gets a unique
         starting value. If None, then chooses automatically based on season-length.
        """

        # parse date information:
        if season_start is None:
            warn("`season_start` was not passed; will assume all groups start in same season.")
            self.start_datetime = None
        else:
            assert timestep_interval is not None, "If passing `season_start` must also pass `timestep_interval`."
            self.start_datetime = np.datetime64(season_start, (timestep_interval, 1))

        self.num_seasons = num_seasons
        self.season_duration = season_duration
        self.datetime_data = None if self.start_datetime is None else np.datetime_data(self.start_datetime)

        # state-elements:
        pad_n = len(str(num_seasons))
        state_elements = ['measured'] + [str(i).rjust(pad_n, "0") for i in range(1, num_seasons)]

        # transitions are placeholder, filled in w/batch
        transitions = dict()
        for i in range(num_seasons):
            current = state_elements[i]
            transitions[current] = {current: None}
            if i > 0:
                prev = state_elements[i - 1]
                transitions[current][prev] = None
                transitions['measured'][prev] = None

        # process-covariance:
        self.log_std_dev = Parameter(-5. * torch.ones(1))

        # initial state:
        if use_fourier_init_param is None:
            use_fourier_init_param = 4 if len(self.state_elements) > 10 else False
        self.K = use_fourier_init_param
        if self.K:
            self.initial_state_mean_params = Parameter(torch.zeros((self.K, 2)))
        else:
            self.initial_state_mean_params = Parameter(torch.zeros(len(self.state_elements) - 1))

        self.initial_state_log_std_dev = Parameter(-5. * torch.ones(1))

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

        # expected for_batch kwargs:
        self.expected_batch_kwargs = ['time']
        if self.start_datetime:
            self.expected_batch_kwargs.append('start_datetimes')

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
            delta = self.get_start_delta(start_datetimes) + time
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

        for i in range(1, self.num_seasons):
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
            delta = np.zeros(shape=(batch_size,), dtype=int)
        else:
            delta = self.get_start_delta(start_datetimes)
        season_shift = np.floor(delta / self.season_duration) % self.num_seasons

        means = []
        for i, shift in enumerate(season_shift.astype(int)):
            if i == 0:
                means.append(mean)
            else:
                means.append(torch.cat([mean[-shift:], mean[:-shift]]))
        means = torch.stack(means)

        # cov:
        # TODO: why ones along diag?
        covs = torch.eye(len(self.state_elements)).expand(batch_size, -1, -1)
        covs[:, 0, 0] = torch.pow(torch.exp(self.initial_state_log_std_dev), 2)

        return means, covs

    def initialize_state_mean(self):
        ns = len(self.state_elements)
        state = Tensor(size=(ns,))
        if self.K:
            # fourier series:
            state[:] = 0.
            season = torch.arange(float(ns))
            for idx in range(self.K):
                k = idx + 1
                state += (self.initial_state_mean_params[idx, 0] * torch.sin(2. * np.pi * k * season / ns))
                state += (self.initial_state_mean_params[idx, 1] * torch.cos(2. * np.pi * k * season / ns))
            # adjust for df:
            state = state - state[1:].mean()
            state[1:] = state[1:] - state[0] / ns
            state[0] = -torch.sum(state[1:])
        else:
            state[1:] = self.initial_state_mean_params
            state[0] = -torch.sum(state[1:])
        return state

    def get_start_delta(self, start_datetimes: np.ndarray) -> np.ndarray:
        assert isinstance(start_datetimes, ndarray), "`start_datetimes` must be a datetime64 numpy.ndarray"
        if self.datetime_data != np.datetime_data(start_datetimes.dtype):
            raise ValueError(f"`start_datetimes` must have same time-unit/step as the `start_datetime` that was passed to "
                             f"seasonal process '{self.id}', which is `{np.datetime_data(self.start_datetime)}`.'")

        td = np.timedelta64(1, self.datetime_data)
        delta = (start_datetimes - self.start_datetime) / td
        return delta.astype(int)

    def set_to_simulation_mode(self):
        super().set_to_simulation_mode()

        self.initial_state_mean_params[:] = 0.
        self.initial_state_mean_params[:, 0] = log(len(self.state_elements)) / 2.
        self.initial_state_log_std_dev[:] = -10.0  # season-effect virtually identical for all groups
        self.log_std_dev[:] = -8.0  # season-effect is very close to stationary
