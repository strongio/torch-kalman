from typing import Union, Sequence, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import ParameterList

from torch_kalman.design import Design, DesignForBatch
from torch_kalman.state_belief import StateBelief, Gaussian
from torch_kalman.state_belief.over_time import StateBeliefOverTime

import numpy as np


class KalmanFilter(torch.nn.Module):
    def __init__(self, design: Design):
        super().__init__()
        self.design = design

        # parameters from design:
        self.design_parameters = ParameterList()
        for param in self.design.parameters():
            self.design_parameters.append(param)

        # the distributional family, implemented by property (default gaussian)
        self._family = None

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

    def initial_state_prediction(self,
                                 input: Union[Tensor, None],
                                 batch_size: Union[int, None] = None,
                                 **kwargs
                                 ) -> StateBelief:
        if input is not None:
            raise NotImplementedError("The default method for `initial_state_prediction` does not take a tensor of "
                                      "predictors; please override this method if you'd like to predict the initial state.")

        means, covs = self.get_block_diag_initial_state(batch_size=batch_size, **kwargs)
        return self.family(means=means, covs=covs)

    def get_block_diag_initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        means = torch.zeros((batch_size, self.state_size))
        covs = torch.zeros((batch_size, self.state_size, self.state_size))

        start = 0
        for process_id, process in self.design.processes.items():
            process_kwargs = {k: kwargs.get(k) for k in process.expected_batch_kwargs}
            process_means, process_covs = process.initial_state(batch_size=batch_size, **process_kwargs)
            end = start + process_means.shape[1]
            means[:, start:end] = process_means
            covs[np.ix_(range(batch_size), range(start, end), range(start, end))] = process_covs
            start = end

        return means, covs

    # noinspection PyShadowingBuiltins
    def forward(self,
                input: Tensor,
                initial_state: Union[Tensor, StateBelief, None] = None,
                **kwargs
                ) -> StateBeliefOverTime:
        """
        :param input: The multivariate time-series to be fit by the kalman-filter. A Tensor where the first dimension
        represents the groups, the second dimension represents the time-points, and the third dimension represents the
        measures.
        :param initial_state: If a StateBelief, this is used as the prediction for time=0. Otherwise, passed to the `input`
        argument of the `initial_state_prediction` method.
        :param kwargs: Other kwargs that will be passed to the `design_for_batch` method.
        :return: A StateBeliefOverTime consisting of one-step-ahead predictions.
        """

        num_groups, num_timesteps, num_measures = input.shape
        if num_measures != self.measure_size:
            raise ValueError(f"This KalmanFilter has {self.measure_size} measurement-dimensions; but the input shape is "
                             f"{(num_groups, num_timesteps, num_measures)} (last dim should be == measure-size).")

        # initial state of the system:
        if isinstance(initial_state, StateBelief):
            state_prediction = initial_state
        else:
            state_prediction = self.initial_state_prediction(input=initial_state, batch_size=num_groups, **kwargs)

        # generate one-step-ahead predictions:
        state_predictions = []
        for t in range(num_timesteps):
            if t > 0:
                # take state-prediction of previous t (now t-1), correct it according to what was actually measured at at t-1
                state_belief = state_prediction.update(obs=input[:, t - 1, :])

                # predict the state for t, from information from t-1
                # F at t-1 is transition *from* t-1 *to* t
                state_prediction = state_belief.predict(F=batch_design.F(), Q=batch_design.Q())

            # compute the design of the kalman filter for this time-point:
            batch_design = self.design_for_batch(batch_size=num_groups, time=t, **kwargs)

            # compute how state-prediction at t translates into measurement-prediction at t
            state_prediction.compute_measurement(H=batch_design.H(), R=batch_design.R())

            # append to output:
            state_predictions.append(state_prediction)

        return self.family.concatenate_over_time(state_beliefs=state_predictions)

    def design_for_batch(self, batch_size: int, time: int, **kwargs) -> DesignForBatch:
        # by overriding this method, child-classes can implement batch-specific changes to design
        return self.design.for_batch(batch_size=batch_size, time=time, **kwargs)

    def forecast(self,
                 state_belief: Union[StateBelief, StateBeliefOverTime],
                 horizon: int,
                 **kwargs) -> StateBeliefOverTime:
        """
        :param state_belief: The output of a call to this KalmanFilter.
        :param horizon: How many timesteps into the future is the forecast?
        :param kwargs: Other arguments passed to this KalmanFilter's `design_for_batch` method. Note: if there are any
        `Seasonal` processes, the `start_datetimes` you pass here will indicate the datetime of horizon=1 (i.e., one
        time-step *after* the most recent data).
        :return: A StateBeliefOverTime consisting of forecasts.
        """

        assert horizon > 0

        if isinstance(state_belief, StateBeliefOverTime):
            state_belief = state_belief.state_beliefs[-1]

        batch_size = state_belief.batch_size

        state_prediction = state_belief
        state_predictions = []
        for h in range(horizon):
            batch_design = self.design_for_batch(batch_size=batch_size, time=h, **kwargs)
            state_prediction = state_prediction.predict(F=batch_design.F(), Q=batch_design.Q())
            state_prediction.compute_measurement(H=batch_design.H(), R=batch_design.R())
            state_predictions.append(state_prediction)

        return self.family.concatenate_over_time(state_beliefs=state_predictions)
