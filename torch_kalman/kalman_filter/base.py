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

    # noinspection PyShadowingBuiltins
    def forward(self,
                input: Any,
                initial_state: Optional[StateBelief] = None,
                progress: Union[tqdm, bool] = False,
                **kwargs) -> StateBeliefOverTime:
        """
        :param input: The multivariate time-series to be fit by the kalman-filter. The exact structure depends on the kalman-
        filter `family`; for most, it is a tensor where the first dimension represents the groups, the second dimension
        represents the time-points, and the third dimension represents the measures.
        :param initial_state: If a StateBelief, this is used as the prediction for time=0; if None then each process
        generates initial values.
        :param progress: Should progress-bar be generated?
        :param kwargs: Other kwargs that will be passed to the `design_for_batch` method.
        :return: A StateBeliefOverTime consisting of one-step-ahead predictions.
        """

        num_groups, num_timesteps, num_measures, *_ = self.family.get_input_dim(input)
        if num_measures != self.measure_size:
            raise ValueError(f"This KalmanFilter has {self.measure_size} measurement-dimensions; but the input shape is "
                             f"{(num_groups, num_timesteps, num_measures)} (3rd dim should == measure-size).")

        design_for_batch = self.design_for_batch(num_groups=num_groups,
                                                 num_timesteps=num_timesteps,
                                                 **kwargs)

        # initial state of the system:
        if initial_state is None:
            state_prediction = self.predict_initial_state(design_for_batch)
        else:
            state_prediction = initial_state

        progress = progress or identity
        if progress is True:
            progress = tqdm
        iterator = progress(range(num_timesteps))

        # generate one-step-ahead predictions:
        state_predictions = []
        for t in iterator:
            if t > 0:
                # take state-prediction of previous t (now t-1), correct it according to what was actually measured at at t-1
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

    def simulate(self,
                 states: Union[StateBeliefOverTime, StateBelief],
                 horizon: int,
                 num_iter: int,
                 progress: bool = False,
                 from_times: Sequence[int] = None,
                 state_to_measured: Optional[Callable] = None,
                 white_noise: Optional[Tuple[Tensor, Tensor]] = None,
                 ntry_diag_incr: int = 1000,
                 **kwargs) -> List[Tensor]:
        """

        :param states: Either the output of the forward pass (a StateBeliefOverTime), or a particular StateBelief.
        :param horizon: The number of timesteps forward to simulate.
        :param num_iter: The number of sim-iterations.
        :param progress: Should progress bar be printed?
        :param from_times: If states is a StateBeliefOverTime, can indicate which times to extract the StateBelief from.
        :param state_to_measured: Optional. A function that takes the StateBeliefOverTime generated from the simulation, and
        converts it into a Tensor of simulated measurements.
        :param white_noise: An optional tuple of tensors, so that the direction of noise can be controlled as a constant
        across sims. Each must have shape (num_groups * num_sims, num_times, ...).
        :param ntry_diag_incr: When simulating from some kalman-filters with low-process variance, the state-belief
        covariance may not be cholesky-decomposible. In this case, we retry the decomposition after adding a small
        value (.000000001) to the diagonal. `ntry_diag_incr` is the number of retries.
        :param kwargs: Further keyword arguments passed to design_for_batch.
        :return: A list of Tensors. Each element of the list is a different sim-iteration.
        """

        assert horizon > 0

        # forecast-from time:
        if from_times is None:
            if isinstance(states, StateBelief):
                initial_state = states
            else:
                # a StateBeliefOverTime was passed, but no from_times, so just pick the last one
                initial_state = states.last_prediction()
        else:
            # from_times will be used to pick the slice
            initial_state = states.state_belief_for_time(from_times)

        initial_state = initial_state.__class__(means=initial_state.means.repeat((num_iter, 1)),
                                                covs=initial_state.covs.repeat((num_iter, 1, 1)),
                                                last_measured=initial_state.last_measured.repeat(num_iter))

        design_for_batch = self.design_for_batch(num_groups=initial_state.num_groups,
                                                 num_timesteps=horizon,
                                                 **kwargs)

        if white_noise is None:
            process_wn, measure_wn = None, None
        else:
            process_wn, measure_wn = white_noise
        trajectories = initial_state.simulate_trajectories(design_for_batch=design_for_batch,
                                                           progress=progress,
                                                           ntry_diag_incr=ntry_diag_incr,
                                                           eps=process_wn)
        if state_to_measured is None:
            sim = trajectories.sample_measurements(eps=measure_wn)
        else:
            sim = state_to_measured(trajectories)

        return torch.chunk(sim, num_iter)

    def forecast(self,
                 states: Union[StateBeliefOverTime, StateBelief],
                 horizon: int,
                 from_times: Optional[Sequence[int]] = None,
                 progress: bool = False,
                 **kwargs) -> StateBeliefOverTime:

        assert horizon > 0

        # forecast-from time:
        if from_times is None:
            if isinstance(states, StateBelief):
                state_prediction = states
            else:
                # a StateBeliefOverTime was passed, but no from_times, so just pick the last one
                state_prediction = states.last_prediction()
        else:
            # from_times will be used to pick the slice
            state_prediction = states.state_belief_for_time(from_times)

        design_for_batch = self.design_for_batch(num_groups=state_prediction.num_groups,
                                                 num_timesteps=horizon,
                                                 **kwargs)

        progress = progress or identity
        if progress is True:
            progress = tqdm
        iterator = progress(range(design_for_batch.num_timesteps))

        forecasts = []
        for t in iterator:
            if t > 0:
                # predict the state for t, from information from t-1
                # F at t-1 is transition *from* t-1 *to* t
                F = design_for_batch.F(t - 1)
                Q = design_for_batch.Q(t - 1)
                state_prediction = state_prediction.predict(F=F, Q=Q)

            # compute how state-prediction at t translates into measurement-prediction at t
            H = design_for_batch.H(t)
            R = design_for_batch.R(t)
            state_prediction.compute_measurement(H=H, R=R)

            # append to output:
            forecasts.append(state_prediction)

        return self.family.concatenate_over_time(state_beliefs=forecasts, design=self.design)
