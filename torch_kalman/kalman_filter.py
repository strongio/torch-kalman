from typing import Optional, Union, Sequence, Any
from warnings import warn

import torch
from torch.nn import Module

from tqdm import tqdm

from torch_kalman.design import Design
from torch_kalman.process import Process
from torch_kalman.state_belief import Gaussian, StateBelief
from torch_kalman.state_belief.over_time import StateBeliefOverTime
from torch_kalman.internals.utils import identity


class KalmanFilter(Module):
    """
    TODO
    """
    family = Gaussian
    design_cls = Design

    def __init__(self, measures: Sequence[str], processes: Sequence[Process], **kwargs):

        super().__init__()
        self.design = self.design_cls(measures=measures, processes=processes, **kwargs)

        # parameters from design:
        self.design_parameters = self.design.param_dict()

        # generally a more reasonable init:
        self.design.process_covariance.set(self.design.process_covariance.create().data / 10.)

    def predict_initial_state(self, design_for_batch: Design) -> 'Gaussian':
        return self.family(
            means=design_for_batch.initial_mean,
            covs=design_for_batch.initial_covariance,
            # we consider this a one-step-ahead prediction, so last measured one step ago:
            last_measured=torch.ones(design_for_batch.num_groups, dtype=torch.int)
        )

    def forward(self,
                input: Any,
                forecast_horizon: Optional[int] = None,
                out_timesteps: Optional[int] = None,
                progress: Union[tqdm, bool] = False,
                initial_prediction: Optional[StateBelief] = None,
                **kwargs) -> StateBeliefOverTime:
        """
        Generate 1-step-ahead predictions.

        :param input: The multivariate time-series to be fit by the kalman-filter. The exact structure depends on the
        kalman-filter `family`; for most, it is a tensor where the first dimension represents the groups, the second
        dimension represents the time-points, and the third dimension represents the measures.
        :param forecast_horizon: Number of timesteps past the end of the input to continue making predictions. Defaults
        to 0. Ignored if `out_timesteps` is specified.
        :param out_timesteps: The number of timesteps to generate predictions for. Sometimes more convenient than
        `forecast_horizon` if predictors are being used, since you can pass `out_timesteps=predictors.shape[1]`
        rather than having to compare the dimensions of the input tensor and the predictor tensor.
        :param progress: Should progress-bar be generated?
        :param initial_prediction: Usually left `None` so that initial predictions are made automatically; in some
        cases case pass a StateBelief generated from a previous prediciton.
        :param kwargs: Other kwargs that will be passed to the kf's `design.for_batch()` method.
        :return: A StateBeliefOverTime consisting of one-step-ahead predictions.
        """
        if input is None:
            if initial_prediction is None:
                raise ValueError("Can only pass input=None if `initial_prediction` is passed.")
            num_groups = initial_prediction.num_groups
            input_num_timesteps = 0
        else:
            num_groups, input_num_timesteps, num_measures, *_ = self.family.get_input_dim(input)
            if num_measures != len(self.design.measures):
                raise ValueError(
                    f"This KalmanFilter has {len(self.design.measures)} measures; but the input shape is "
                    f"{(num_groups, input_num_timesteps, num_measures)} (3rd dim should == measure-size)."
                )

        # times
        if out_timesteps is None:
            if forecast_horizon is None:
                out_timesteps = input_num_timesteps
            else:
                assert forecast_horizon >= 0
                out_timesteps = input_num_timesteps + forecast_horizon
        else:
            if forecast_horizon is not None:
                warn("`out_timesteps` was specified so `forecast_horizon` will be ignored.")

        progress = progress or identity
        if progress is True:
            progress = tqdm
        times = progress(range(out_timesteps))

        design_for_batch = self.design.for_batch(num_groups=num_groups, num_timesteps=out_timesteps, **kwargs)

        # initial state of the system:
        if initial_prediction is None:
            state_prediction = self.predict_initial_state(design_for_batch)
        else:
            state_prediction = initial_prediction.copy()

        # generate predictions:
        state_predictions = []
        for t in times:
            if t > 0:
                if input is None:
                    state_belief = state_prediction.copy()
                else:
                    # take state-pred of previous t (now t-1), correct it according to what was actually measured at t-1
                    state_belief = state_prediction.update_from_input(input, time=t - 1)

                # predict the state for t, from information from t-1
                # F at t-1 is transition *from* t-1 *to* t
                F = design_for_batch.F(t - 1)
                Q = design_for_batch.Q(t - 1)
                state_prediction = state_belief.predict(F=F, Q=Q)

            # compute how state-prediction at t translates into measurement-prediction at t
            H = design_for_batch.H(t)
            R = design_for_batch.R(t)
            state_prediction.compute_measurement(H=H, R=R)

            # append to output:
            state_predictions.append(state_prediction)

        return self.family.concatenate_over_time(state_beliefs=state_predictions, design=self.design)

    def smooth(self, states: StateBeliefOverTime):
        raise NotImplementedError
