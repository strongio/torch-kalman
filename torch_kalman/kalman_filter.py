"""
Base class for torch.nn.Modules that generate predictions with the Kalman-filtering algorithm.
"""

from typing import Optional, Union, Sequence
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
    :cvar family: A subclass of `StateBelief`, representing the distribution being predicted. In this base class this is
     `Gaussian`. `torch-kalman` also includes `CensoredGaussian`.
    :cvar design_cls: The class that receives the `measures` and `processes`. In this base class this is `Design`.
    """
    family = Gaussian
    design_cls = Design

    def __init__(self,
                 measures: Sequence[str],
                 processes: Sequence[Process],
                 measure_var_predict: Sequence[torch.nn.Module] = (),
                 process_var_predict: Sequence[torch.nn.Module] = (),
                 **kwargs):
        """
        :param processes: Processes
        :param measures: Measure-names
        :param measure_var_predict: If a torch.nn.Module (or a list of these) is passed, this will be used to predict
        the measure-variance. Then when calling your KalmanFilter, the input to this module can be passed using the
        keyword-arguments to the measure-var-predictor's `forward()` method (e.g. if you pass a single module
        `MyVarPredictModule`, with `forward()` that takes `input`, you would call your KalmanFilter with
        `measure_var_nn0__input=[predictor-tensor]`). In addition to passing a torch.nn.Module, you can pass a tuple
        with the format `('alias',alias_args)`. There are currently two supported aliases. The 'per_group' alias (e.g.
        `('per_group',(number_of_groups,))`) allows a separate set of variance-adjustment parameters for each group
        passed to the KalmanFilter. The 'seasonal' alias (e.g. `('seasonal',{'K':2,'period':'yearly','dt_unit':'W'})`)
        allows the variance to vary in a seasonal pattern, as implemented by `FourierSeasonNN`. Multiple modules and
        aliases can be passed as a list (e.g. `[('per_group',1), ('seasonal',yearly_args), ('seasonal',weekly_args)]`).
        :param process_var_predict: See `measure_var_predict`.
        :param kwargs: For child-classes; other keyword arguments to pass to `KalmanFilter.design_cls`.
        """

        super().__init__()
        self.design = self.design_cls(
            measures=measures,
            processes=processes,
            measure_var_predict=measure_var_predict,
            process_var_predict=process_var_predict,
            **kwargs
        )

        # parameters from design:
        self.design_parameters = self.design.param_dict()

        # generally a more reasonable init:
        self.design.process_covariance.set(self.design.process_covariance.create().data / 10.)

    def forward(self,
                *args,
                forecast_horizon: Optional[int] = None,
                out_timesteps: Optional[int] = None,
                progress: Union[tqdm, bool] = False,
                initial_prediction: Optional[StateBelief] = None,
                **kwargs) -> StateBeliefOverTime:
        """
        Generate 1-step-ahead predictions.

        :param args: For this base class, a single Tensor containing a batch of time-series with dims
            (group, time, measure). Child classes using a different :code:`KalmanFilter.family` may accept additional
            args (e.g. :code:`CensoredGaussian` requires three arguments: the time-series, the lower censoring limits,
            and the upper-censoring limits).
        :param forecast_horizon: Number of timesteps past the end of the input to continue making predictions. Defaults
        to 0. Ignored if `out_timesteps` is specified.
        :param out_timesteps: The number of timesteps to generate predictions for. Sometimes more convenient than
        `forecast_horizon` if predictors are being used, since you can pass `out_timesteps=predictors.shape[1]`
        rather than having to compare the dimensions of the input tensor and the predictor tensor.
        :param progress: Should progress-bar be generated?
        :param initial_prediction: Usually left `None` so that initial predictions are made automatically; in some
        cases case you might pass a StateBelief generated from a previous prediciton.
        :param kwargs: Other kwargs that will be passed to the kf's `design.for_batch()` method, which in turn passes
        them to each process (or to the NNs specified in `*_var_predict`). Sometimes, processes might share keyword-
        argument names but you want to pass different arguments to them -- for example, if you have two processes that
        take `predictors` but you want to pass a different predictor matrix to each. In this case you can disambiguate
        using sklearn-style double-underscoring: `process1__predictors` and `process2__predictors`.
        :return: A StateBeliefOverTime consisting of one-step-ahead predictions.
        """
        if not args:
            if initial_prediction is None:
                raise ValueError("No input tensor was passed, so must pass `initial_prediction`.")
            num_groups = initial_prediction.num_groups
            input_num_timesteps = 0
        else:
            num_groups, input_num_timesteps, num_measures, *_ = args[0].shape
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
            state_prediction = self._predict_initial_state(design_for_batch)
        else:
            state_prediction = initial_prediction.copy()

        # generate predictions:
        state_predictions = []
        for t in times:
            if t > 0:
                if not args:
                    state_belief = state_prediction.copy()
                else:
                    # take state-pred of previous t (now t-1), correct it according to what was actually measured at t-1
                    state_belief = state_prediction.update(*args, time=t - 1)

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

    def _predict_initial_state(self, design_for_batch: Design) -> 'Gaussian':
        return self.family(
            means=design_for_batch.initial_mean,
            covs=design_for_batch.initial_covariance,
            # we consider this a one-step-ahead prediction, so last measured one step ago:
            last_measured=torch.ones(design_for_batch.num_groups, dtype=torch.int)
        )
