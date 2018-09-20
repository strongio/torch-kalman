from typing import Generator

import torch
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process


class Season(Process):
    def __init__(self, id: str, num_seasons: int):
        """
        Seasonal process where seasons have no duration (i.e., they increment every timestep)

        :param id: The id of the process
        :param num_seasons: The number of seasons
        """
        # state-elements:
        pad_n = len(str(num_seasons))
        state_elements = ['measured']
        for i in range(1, num_seasons):
            state_elements.append(str(i).rjust(pad_n, "0"))

        # transitions:
        transitions = {}

        # the first element is special:
        transitions['measured'] = dict.fromkeys(state_elements, -1.0)
        transitions['measured'].pop(state_elements[-1])  # all but the last

        # the rest of the elements:
        for i in range(1, num_seasons):
            current = state_elements[i]
            prev = state_elements[i - 1]
            transitions[current] = {prev: 1.0}

        # process-covariance:
        self.log_std_dev = Parameter(data=torch.randn(1))

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.log_std_dev

    def covariance(self) -> Covariance:
        state_size = len(self.state_elements)
        cov = Covariance(size=(state_size, state_size))
        cov[:] = 0.
        cov[0, 0] = torch.pow(torch.exp(self.log_std_dev), 2)
        return cov

    @property
    def measurable_state(self) -> str:
        return 'measured'
