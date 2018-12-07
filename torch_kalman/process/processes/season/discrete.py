from math import log
from typing import Optional, Union, Generator, Dict, Sequence, Tuple

import numpy as np
import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process.utils.fourier import fourier_tensor
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.processes.season.base import DateAware
from torch_kalman.process.utils.transition import Transition
from torch_kalman.utils import split_flat


class Season(DateAware):
    def __init__(self,
                 id: str,
                 seasonal_period: int,
                 season_duration: int = 1,
                 decay: Union[bool, Tuple[float, float]] = False,
                 use_fourier_init_param: Optional[int] = None,
                 *args, **kwargs):
        """
        Process representing discrete seasons.

        :param id: Unique name for this process
        :param seasonal_period: The number of seasons (e.g. 7 for day_in_week).
        :param season_duration: The length of each season, default 1 time-step.
        :param season_start: A string that can be parsed into a datetime by `numpy.datetime64`. This is when the season
        starts, which is useful to specify if season boundaries are actually meaningful, and is important to specify if
        different groups in your dataset start on different dates.
        :param decay: Analogous to dampening a trend -- the state will revert to zero as we get further from the last
        observation. This can be useful if two processes are capturing the same seasonal pattern: one can be more flexible,
        but with decay have a tendency to revert to zero, while the other is less variable but extrapolates into the future.
        :param use_fourier_init_param: This determines how the *initial* state-means are parameterized. For longer seasons,
        we may not want a free parameter for each individual season. If an integer, then uses a fourier-series for the
        parameterization, with that integer's (*2) degrees of freedom. If False, then there are seasonal_period - 1 unique
         initial values (-1 b/c constrained to sum to zero). Default (None) chooses automatically (fourier if >12 period).
        """

        self.seasonal_period = seasonal_period
        self.season_duration = season_duration

        # state-elements:
        pad_n = len(str(seasonal_period))
        state_elements = ['measured'] + [str(i).rjust(pad_n, "0") for i in range(1, seasonal_period)]

        #
        transitions = self.initialize_transitions(state_elements=state_elements)

        # process-covariance:
        self.log_std_dev = Parameter(torch.randn(1) - 3.)

        # initial state:
        if use_fourier_init_param is None:
            use_fourier_init_param = 4 if self.seasonal_period > 12 else False
        self.K = use_fourier_init_param
        if self.K:
            self.initial_state_mean_params = Parameter(torch.randn((self.K, 2)))
        else:
            self.initial_state_mean_params = Parameter(torch.randn(self.seasonal_period - 1))

        self.initial_state_log_std_dev = Parameter(torch.randn(1) - 3.)

        if decay:
            assert not isinstance(decay, bool), "decay should be floats of bounds (or False for no decay)"
            self.decay = Transition(*decay)
        else:
            self.decay = None

        # super:
        super().__init__(id=id,
                         state_elements=state_elements,
                         transitions=transitions,
                         *args, **kwargs)

        # writing transition-matrix is slow, no need to do it repeatedly:
        self.transition_cache = {}

    @property
    def measured_name(self) -> str:
        return self.state_elements[0]

    def add_measure(self,
                    measure: str,
                    state_element: str = 'measured',
                    value: Union[float, Tensor, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.log_std_dev
        yield self.initial_state_log_std_dev
        yield self.initial_state_mean_params
        if self.decay is not None:
            yield self.decay.parameter

    def covariance(self) -> Covariance:
        state_size = len(self.state_elements)
        cov = torch.zeros(size=(state_size, state_size), device=self.device)
        cov[0, 0] = torch.pow(torch.exp(self.log_std_dev), 2)
        return cov

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  start_datetimes: Optional[np.ndarray] = None) -> ProcessForBatch:
        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps, start_datetimes=start_datetimes)
        delta = self.get_delta(for_batch.num_groups, for_batch.num_timesteps, start_datetimes=start_datetimes)

        in_transition = (delta % self.season_duration) == (self.season_duration - 1)

        transitions = {}
        transitions['to_next_state'] = torch.from_numpy(in_transition.astype('float32'))
        transitions['to_self'] = 1 - transitions['to_next_state']
        transitions['to_measured'] = -transitions['to_next_state']
        transitions['from_measured_to_measured'] = torch.from_numpy(np.where(in_transition, -1., 1.).astype('float32'))
        for k in transitions.keys():
            transitions[k] = split_flat(transitions[k], dim=1)
            if self.decay is not None:
                transitions[k] = [x * self.decay.value for x in transitions[k]]

        # this is convoluted, but the idea is to manipulate the transitions so that we use one less degree of freedom than
        # the number of seasons, by having the 'measured' state be equal to -sum(all others)
        for i in range(1, len(self.state_elements)):
            current = self.state_elements[i]
            prev = self.state_elements[i - 1]

            if prev == self.measured_name:  # measured requires special-case
                to_measured = transitions['from_measured_to_measured']
            else:
                to_measured = transitions['to_measured']

            for_batch.set_transition(from_element=prev, to_element=current, values=transitions['to_next_state'])
            for_batch.set_transition(from_element=prev, to_element=self.measured_name, values=to_measured)

            # from state to itself:
            for_batch.set_transition(from_element=current, to_element=current, values=transitions['to_self'])

        return for_batch

    def initial_state(self):
        if self.K:
            # fourier series:
            season = torch.arange(float(self.seasonal_period), device=self.device)
            fourier_mat = fourier_tensor(time=season, seasonal_period=self.seasonal_period, K=self.K)
            init_mean = torch.sum(fourier_mat * self.initial_state_mean_params.expand_as(fourier_mat), dim=(1, 2))
            # adjust for df:
            init_mean = init_mean - init_mean[1:].mean()
            init_mean[1:] = init_mean[1:] - init_mean[0] / self.seasonal_period
            init_mean[0] = -torch.sum(init_mean[1:])
        else:
            init_mean = Tensor(size=(self.seasonal_period,), device=self.device)
            init_mean[1:] = self.initial_state_mean_params
            init_mean[0] = -torch.sum(init_mean[1:])

        init_cov = torch.eye(len(self.state_elements), device=self.device)
        init_cov[0, 0] = torch.pow(torch.exp(self.initial_state_log_std_dev), 2)

        return init_mean, init_cov

    def initial_state_for_batch(self, num_groups: int, start_datetimes: Optional[np.ndarray] = None):
        delta = self.get_delta(num_groups, 1, start_datetimes=start_datetimes).squeeze(1)

        init_mean, init_cov = self.initial_state()
        season_shift = (np.floor(delta / self.season_duration) % self.seasonal_period).astype('int')

        means = [torch.cat([init_mean[-shift:], init_mean[:-shift]]) for shift in season_shift]
        means = torch.stack(means)

        covs = init_cov.expand(num_groups, -1, -1)

        return means, covs

    @staticmethod
    def initialize_transitions(state_elements: Sequence[str]) -> Dict[str, Dict[str, None]]:
        measured_name = state_elements[0]
        # transitions are placeholder, filled in w/batch
        transitions = {}
        for i in range(len(state_elements)):
            current = state_elements[i]
            transitions[current] = {current: None}
            if i > 0:
                prev = state_elements[i - 1]
                transitions[current][prev] = None
                transitions[measured_name][prev] = None

        return transitions

    def set_to_simulation_mode(self):
        super().set_to_simulation_mode()

        self.initial_state_mean_params[:] = 0.
        self.initial_state_mean_params[:, 0] = log(len(self.state_elements)) / 2.
        self.initial_state_log_std_dev[:] = -10.0  # season-effect virtually identical for all groups
        self.log_std_dev[:] = -8.0  # season-effect is very close to stationary
