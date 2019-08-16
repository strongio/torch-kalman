from typing import Type, Optional, Callable, List, Union, Tuple, Sequence, Any

import torch
from torch import Tensor

from tqdm import tqdm

from torch_kalman.design import Design
from torch_kalman.design.for_batch import DesignForBatch
from torch_kalman.process import Process
from torch_kalman.state_belief import Gaussian, StateBelief
from torch_kalman.state_belief.over_time import StateBeliefOverTime
from torch_kalman.utils import identity


class KalmanFilter(torch.nn.Module):
    family: Type[StateBelief] = Gaussian
    design_cls = Design

    def __init__(self,
                 measures: Sequence[str],
                 processes: Sequence[Process],
                 **kwargs):

        super().__init__()
        self.design = self.design_cls(measures=measures, processes=processes, **kwargs)

        # parameters from design:
        self.design_parameters = self.design.param_dict()

        # the StateBelief family, implemented by property (default gaussian)
        self._family = None

        self.to(device=self.design.device)

    @property
    def measure_size(self) -> int:
        return self.design.measure_size

    def predict_initial_state(self, design_for_batch: DesignForBatch) -> 'Gaussian':
        return self.family(means=design_for_batch.initial_mean,
                           covs=design_for_batch.initial_covariance,
                           # we consider this a one-step-ahead prediction, so last measured one step ago:
                           last_measured=torch.ones(design_for_batch.num_groups, dtype=torch.int))

    def design_for_batch(self,
                         num_groups: int,
                         num_timesteps: int,
                         **kwargs) -> DesignForBatch:
        return self.design.for_batch(num_groups=num_groups, num_timesteps=num_timesteps, **kwargs)

    def forward(self,
                input: Any,
                initial_prediction: Optional[StateBelief] = None,
                forecast_horizon: int = 0,
                progress: Union[tqdm, bool] = False,
                **kwargs) -> StateBeliefOverTime:
        """
        :param input: The multivariate time-series to be fit by the kalman-filter. The exact structure depends on the kalman-
        filter `family`; for most, it is a tensor where the first dimension represents the groups, the second dimension
        represents the time-points, and the third dimension represents the measures.
        :param initial_prediction: If a StateBelief, this is used as the prediction for time=0; if None then each process
        generates initial values.
        :param forecast_horizon: Number of timesteps past the end of the input to continue making predictions
        :param progress: Should progress-bar be generated?
        :param kwargs: Other kwargs that will be passed to the `design_for_batch` method.
        :return: A StateBeliefOverTime consisting of one-step-ahead predictions.
        """

        num_groups, num_timesteps_input, num_measures, *_ = self.family.get_input_dim(input)
        if num_measures != self.measure_size:
            raise ValueError(f"This KalmanFilter has {self.measure_size} measurement-dimensions; but the input shape is "
                             f"{(num_groups, num_timesteps_input, num_measures)} (3rd dim should == measure-size).")

        assert forecast_horizon >= 0
        num_timesteps = num_timesteps_input + forecast_horizon

        design_for_batch = self.design_for_batch(num_groups=num_groups,
                                                 num_timesteps=num_timesteps,
                                                 **kwargs)

        # initial state of the system:
        if initial_prediction is None:
            state_prediction = self.predict_initial_state(design_for_batch)
        else:
            state_prediction = initial_prediction.copy()

        progress = progress or identity
        if progress is True:
            progress = tqdm
        times = progress(range(num_timesteps))

        # generate one-step-ahead predictions:
        state_predictions = []
        for t in times:
            if t > 0:
                # take state-prediction of previous t (now t-1), correct it according to what was actually measured at t-1
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
