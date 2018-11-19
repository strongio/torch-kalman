from typing import Union, Iterable, Optional

import torch
from torch import Tensor
from torch.nn import ParameterList

from torch_kalman.design import Design
from torch_kalman.process import Process
from torch_kalman.state_belief import StateBelief, Gaussian
from torch_kalman.state_belief.over_time import StateBeliefOverTime


class KalmanFilter(torch.nn.Module):
    def __init__(self,
                 processes: Iterable[Process],
                 measures: Iterable[str],
                 device: Optional[torch.device] = None):
        super().__init__()
        self.design = Design(processes=processes, measures=measures, device=device)

        # parameters from design:
        self.design_parameters = ParameterList()
        for param in self.design.parameters():
            self.design_parameters.append(param)

        # the distributional family, implemented by property (default gaussian)
        self._family = None

        self.to(device=self.design.device)

    @property
    def family(self):
        if self._family is None:
            self._family = Gaussian
        return self._family

    @property
    def state_size(self) -> int:
        return self.design.state_size

    @property
    def measure_size(self) -> int:
        return self.design.measure_size

    # noinspection PyShadowingBuiltins
    def forward(self,
                input: Tensor,
                initial_state: Union[StateBelief, None] = None,
                **kwargs
                ) -> StateBeliefOverTime:
        """
        :param input: The multivariate time-series to be fit by the kalman-filter. A Tensor where the first dimension
        represents the groups, the second dimension represents the time-points, and the third dimension represents the
        measures.
        :param initial_state: If a StateBelief, this is used as the prediction for time=0; if None then each process
        generates initial values.
        :param kwargs: Other kwargs that will be passed to the `design_for_batch` method.
        :return: A StateBeliefOverTime consisting of one-step-ahead predictions.
        """

        num_groups, num_timesteps, num_measures = input.shape
        if num_measures != self.measure_size:
            raise ValueError(f"This KalmanFilter has {self.measure_size} measurement-dimensions; but the input shape is "
                             f"{(num_groups, num_timesteps, num_measures)} (last dim should == measure-size).")

        design_for_batch = self.design.for_batch(input=input, **kwargs)

        # initial state of the system:
        if initial_state is None:
            state_prediction = self.family(*design_for_batch.get_block_diag_initial_state(num_groups=num_groups))
        else:
            state_prediction = initial_state

        # generate one-step-ahead predictions:
        state_predictions = []
        for t in range(num_timesteps):
            if t > 0:
                # take state-prediction of previous t (now t-1), correct it according to what was actually measured at at t-1
                state_belief = state_prediction.update(obs=input[:, t - 1, :])

                # predict the state for t, from information from t-1
                # F at t-1 is transition *from* t-1 *to* t
                F = design_for_batch.F[:, t - 1, :, :]
                Q = design_for_batch.Q[:, t - 1, :, :]
                state_prediction = state_belief.predict(F=F, Q=Q)

            # compute how state-prediction at t translates into measurement-prediction at t
            H = design_for_batch.H[:, t, :, :]
            R = design_for_batch.R[:, t, :, :]
            state_prediction.compute_measurement(H=H, R=R)

            # append to output:
            state_predictions.append(state_prediction)

        return self.family.concatenate_over_time(state_beliefs=state_predictions, design=self.design)
