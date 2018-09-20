import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import ParameterList, Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.design import Design, DesignForBatch
from torch_kalman.state_belief import StateBelief, Gaussian


class KalmanFilter(torch.nn.Module):
    def __init__(self, design: Design) -> None:
        super().__init__()
        self.design = design

        # initial state:
        state_size = self.design.state_size()
        self.init_state_mean = Parameter(torch.randn(state_size))
        self.init_state_cholesky_log_diag = Parameter(data=torch.randn(state_size))
        self.init_state_cholesky_off_diag = Parameter(data=torch.randn(int(state_size * (state_size - 1) / 2)))

        # parameters from design:
        self.design_parameters = ParameterList()
        for param in self.design.parameters():
            self.design_parameters.append(param)

        self._state_belief = None

    @property
    def state_belief(self):
        if self._state_belief is None:
            self._state_belief = Gaussian
        return self._state_belief

    def initial_state_prediction(self, input) -> StateBelief:
        covs = Covariance.from_log_cholesky(log_diag=self.init_state_cholesky_log_diag,
                                            off_diag=self.init_state_cholesky_off_diag)
        return self.state_belief(means=self.init_state_mean.expand(len(input), -1),
                                 covs=covs.expand(len(input), -1, -1))

    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Distribution:
        """
        :param input: A group X time X dimension Tensor.
        :return: Tuple with:
            Mean: A groups X time X dimension x 1 Tensor
            Cov: A groups X time X dimension X dimension Tensor
        """

        num_groups, num_timesteps, num_dims_all = input.shape

        state_prediction = self.initial_state_prediction(input)

        state_predictions = []
        for t in range(num_timesteps):
            if t > 0:
                # take state-prediction of previous t (now t-1), correct it according to what was actually measured at at t-1
                state_belief = state_prediction.update(obs=input[:, t - 1, :])

                # predict the state for t, from information from t-1
                # F at t-1 is transition *from* t-1 *to* t
                state_prediction = state_belief.predict(F=batch_design.F(), Q=batch_design.Q())

            # compute the design of the kalman filter for this timestep:
            batch_design = self.design_for_batch(input[:, t, :])

            # compute how state-prediction at t translates into measurement-prediction at t
            state_prediction.measure_state(H=batch_design.H(), R=batch_design.R())

            # append to output:
            state_predictions.append(state_prediction)

        return self.state_belief.concatenate_measured_states(state_beliefs=state_predictions)

    def design_for_batch(self, obs: Tensor) -> DesignForBatch:
        # by overriding this method, child-classes can implement batch-specific changes to design
        return self.design.for_batch(batch_size=len(obs))
