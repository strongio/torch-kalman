from typing import Union

import torch
from torch import Tensor
from torch.nn import ParameterList, Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.design import Design, DesignForBatch
from torch_kalman.state_belief import StateBelief, Gaussian, StateBeliefOverTime


class KalmanFilter(torch.nn.Module):
    def __init__(self, design: Design) -> None:
        super().__init__()
        self.design = design

        # initial state:
        ns = self.state_size
        self.init_state_mean = Parameter(torch.randn(ns))
        self.init_state_cholesky_log_diag = Parameter(data=torch.randn(ns))
        self.init_state_cholesky_off_diag = Parameter(data=torch.randn(int(ns * (ns - 1) / 2)))

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

    def initial_state_prediction(self, input: Union[Tensor, None], batch_size: Union[int, None] = None) -> StateBelief:
        if input is not None:
            raise NotImplementedError("The default method for `initial_state_prediction` does not take a tensor of "
                                      "predictors; please override this method if you'd like to predict the initial state.")
        covs = Covariance.from_log_cholesky(log_diag=self.init_state_cholesky_log_diag,
                                            off_diag=self.init_state_cholesky_off_diag)
        return self.family(means=self.init_state_mean.expand(batch_size, -1),
                           covs=covs.expand(batch_size, -1, -1))

    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor, initial_state: Union[Tensor, StateBelief, None] = None) -> StateBeliefOverTime:
        num_groups, num_timesteps, num_dims_all = input.shape
        if num_dims_all < self.measure_size:
            raise ValueError(f"The design of this KalmanFilter expects a state-size of {self.measure_size}; but the input "
                             f"shape is {(num_groups, num_timesteps, num_dims_all)} (last dim should be >= measure-size).")

        # initial state of the system:
        if isinstance(initial_state, StateBelief):
            state_prediction = initial_state
        else:
            state_prediction = self.initial_state_prediction(input=initial_state, batch_size=num_groups)

        # generate one-step-ahead predictions:
        state_predictions = []
        for t in range(num_timesteps):
            if t > 0:
                # take state-prediction of previous t (now t-1), correct it according to what was actually measured at at t-1
                measurements = input[:, t - 1, :self.measure_size]
                state_belief = state_prediction.update(obs=measurements)

                # predict the state for t, from information from t-1
                # F at t-1 is transition *from* t-1 *to* t
                state_prediction = state_belief.predict(F=batch_design.F(), Q=batch_design.Q())

            # compute the design of the kalman filter for this timestep:
            batch_design = self.design_for_batch(input[:, t, :])

            # compute how state-prediction at t translates into measurement-prediction at t
            state_prediction.compute_measurement(H=batch_design.H(), R=batch_design.R())

            # append to output:
            state_predictions.append(state_prediction)

        return self.family.concatenate_over_time(state_beliefs=state_predictions)

    def design_for_batch(self, input: Tensor) -> DesignForBatch:
        # by overriding this method, child-classes can implement batch-specific changes to design
        return self.design.for_batch(batch_size=len(input))
