from math import floor

import torch
from torch.autograd import Variable

from torch_kalman.utils.torch_utils import Param0


class SeasonNN(torch.nn.Module):
    def __init__(self, period, duration):
        super().__init__()
        if isinstance(period, float):
            assert period.is_integer()
        if isinstance(duration, float):
            assert duration.is_integer()
        self.period = int(period)
        self.duration = int(duration)

    def forward(self, time):
        raise NotImplementedError()


class InitialSeasonStateNN(SeasonNN):
    def __init__(self, period, duration):
        super().__init__(period=period, duration=duration)
        # if we have N seasons, we have N-1 degrees of freedom. the first is always constrained to zero
        self.initial_state_params = Param0(period - 1)

    def forward(self, time_start):
        bs = time_start.data.shape[0]

        # the first season is constrained to be -sum(the-rest) s.t. the seasons sum to zero
        initial_states = Variable(torch.zeros(self.period))
        initial_states[1:] = self.initial_state_params
        initial_states[0] = -torch.sum(initial_states[1:])

        out = Variable(torch.zeros(bs, self.period))
        for i, init_time in enumerate(time_start.data.tolist()):
            # start-time => start-season
            init_season = int(floor(init_time / self.duration))

            # season-indices for start-season thru period:
            idx = [x % self.period for x in range(init_season, init_season + self.period)]

            # the first season is simply -sum(the_rest)
            out[i, :] = initial_states[idx]

        return out


class SeasonDurationNN(SeasonNN):
    def __init__(self, period, duration, season_start):
        super().__init__(period=period, duration=duration)
        if isinstance(season_start, float):
            assert season_start.is_integer()
        self.season_start = season_start

    def forward(self, time_start):
        bs = time_start.data.shape[0]

        out = {key: torch.zeros(bs) for key in ('to_self', 'to_first', 'to_next', 'from_first_to_first')}
        in_transition = (torch.fmod(time_start, self.duration) == self.season_start).data
        out['to_self'][~in_transition] = 1.
        out['to_first'][in_transition] = -1.
        out['to_next'][in_transition] = 1.
        out['from_first_to_first'][in_transition] = -1.
        out['from_first_to_first'][~in_transition] = 1.

        return {k: Variable(v) for k, v in out.items()}
