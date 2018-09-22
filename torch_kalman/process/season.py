from typing import Generator, Union, Tuple, Optional
from warnings import warn

import torch

import numpy as np

from numpy.core.multiarray import ndarray
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process, ProcessForBatch


class Season(Process):
    def __init__(self,
                 id: str,
                 num_seasons: int,
                 season_duration: int = 1,
                 start_datetime: Optional[np.datetime64] = None):

        # parse date information:
        if start_datetime is None:
            warn("`start_datetime` was not passed; will assume all groups start in same season.")
        else:
            assert isinstance(start_datetime, np.datetime64), "`start_datetime` must be a `np.datetime64`."

        self.num_seasons = num_seasons
        self.season_duration = season_duration
        self.start_datetime = start_datetime
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
        self.log_std_dev = Parameter(data=torch.randn(1))

        # initial state:
        self.initial_state_mean_params = Parameter(torch.randn(num_seasons - 1))
        self.initial_state_log_std_dev = Parameter(torch.randn(1))

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

        # expected for_batch kwargs:
        self.expected_batch_kwargs = ['time']
        if self.start_datetime:
            self.expected_batch_kwargs.append('start_datetimes')

        # writing transition-matrix is slow, no need to do it repeatedly:
        self.transition_cache = {}

    def initial_state(self,
                      batch_size: int,
                      start_datetimes: Optional[ndarray] = None,
                      time: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        ns = len(self.state_elements)

        # mean:
        mean = Tensor(size=(ns,))
        mean[1:] = self.initial_state_mean_params
        mean[0] = -torch.sum(mean[1:])

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
        covs = torch.eye(ns).expand(batch_size, -1, -1)
        covs[:, 0, 0] = torch.pow(torch.exp(self.initial_state_log_std_dev), 2)

        return means, covs

    def add_measure(self,
                    measure: str,
                    state_element: str = 'measured',
                    value: Union[float, Tensor, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.log_std_dev
        yield self.initial_state_mean_params
        yield self.initial_state_log_std_dev

    def covariance(self) -> Covariance:
        state_size = len(self.state_elements)
        cov = Covariance(size=(state_size, state_size))
        cov[:] = 0.
        cov[0, 0] = torch.pow(torch.exp(self.log_std_dev), 2)
        return cov

    def for_batch(self,
                  batch_size: int,
                  time: Optional[int] = None,
                  start_datetimes: Optional[ndarray] = None) -> ProcessForBatch:

        if start_datetimes is None:
            if self.start_datetime:
                raise ValueError("`start_datetimes` argument required.")
            delta = np.ones(shape=(batch_size,), dtype=int) * time
        else:
            delta = self.get_start_delta(start_datetimes) + time
        in_transition = (delta % self.season_duration) == (self.season_duration - 1)

        key = in_transition.tostring()
        if key not in self.transition_cache.keys():
            self.transition_cache[key] = self.make_batch_with_transitions(in_transition)

        return self.transition_cache[key]

    def make_batch_with_transitions(self, in_transition: np.ndarray) -> ProcessForBatch:
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

        return for_batch

    def get_start_delta(self, start_datetimes: np.ndarray) -> np.ndarray:
        assert isinstance(start_datetimes, ndarray), "`start_datetimes` must be a datetime64 numpy.ndarray"
        if self.datetime_data != np.datetime_data(start_datetimes.dtype):
            raise ValueError(f"`start_datetimes` must have same time-unit/step as the `start_datetime` that was passed to "
                             f"seasonal process '{self.id}', which is `{np.datetime_data(self.start_datetime)}`.'")

        td = np.timedelta64(1, self.datetime_data)
        delta = (start_datetimes - self.start_datetime) / td
        return delta.astype(int)
