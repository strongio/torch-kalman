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

    def check_input(self, input):
        if len(input.data.shape) > 2:
            raise ValueError("`SeasonNN` expected max. two-dimensional input.")
        elif len(input.data.shape) == 2:
            if input.data.shape[1] == 1:
                input = torch.squeeze(input, 1)
            else:
                raise ValueError("`SeasonNN` received a two-dimensional input where the 2nd dimension wasn't singular.")
        return input

    def forward(self, time):
        raise NotImplementedError()


class InitialSeasonStateNN(SeasonNN):
    def __init__(self, period, duration):
        super().__init__(period, duration)
        # if we have N seasons, we have N-1 degrees of freedom. the first is always constrained to zero
        self.initial_state_params = Param0(period - 1)

    def forward(self, time):
        time = self.check_input(time)

        # the last season doesn't get a parameter:
        initial_states = Variable(torch.zeros(self.period))
        initial_states[:-1] = self.initial_state_params

        out = []
        for init_time in time.data.tolist():
            # convert time each group starts on to season each group starts on:
            init_season = int(floor(init_time / self.duration))
            # add a sequence for the remaining seasons, using modulus operator:
            idx = [x % self.period for x in range(init_season, init_season + self.period)]
            # index initial states:
            out.append(initial_states[idx][None, :])

        return torch.cat(out, 0)


class SeasonDurationNN(SeasonNN):
    def __init__(self, period, duration):
        super().__init__(period, duration)

    def forward(self, time):
        time = self.check_input(time)
        bs = time.data.shape[0]

        out = {key: torch.zeros(bs) for key in ('to_self', 'to_first', 'to_next', 'from_first_to_first')}

        for i, t in enumerate(time.data.tolist()):
            if (t % self.duration) == 0:
                # transition:
                out['to_self'][i] = 0.
                out['to_first'][i] = -1.
                out['to_next'][i] = 1.
                out['from_first_to_first'][i] = -1
            else:
                # inside a season:
                out['to_self'][i] = 1.
                out['to_first'][i] = 0.
                out['to_next'][i] = 0.
                out['from_first_to_first'][i] = 1.

        return {k: Variable(v) for k, v in out.items()}
