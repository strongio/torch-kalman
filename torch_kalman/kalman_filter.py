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
                n_step: int = 1,
                progress: Union[tqdm, bool] = False,
                initial_prediction: Optional[StateBelief] = None,
                **kwargs) -> StateBeliefOverTime:
        """
        Generate n-step-ahead predictions.

        :param args: For this base class, a single Tensor containing a batch of time-series with dims
        (group, time, measure). Child classes using a different :code:`KalmanFilter.family` may accept additional
        args (e.g. :code:`CensoredGaussian` requires three arguments: the time-series, the lower censoring limits,
        and the upper-censoring limits).
        :param forecast_horizon: Number of timesteps past the end of the input to continue making predictions. Defaults
        to 0. Ignored if `out_timesteps` is specified.
        :param out_timesteps: The number of timesteps to generate predictions for. Sometimes more convenient than
        `forecast_horizon` if predictors are being used, since you can pass `out_timesteps=predictors.shape[1]`
        rather than having to compare the dimensions of the input tensor and the predictor tensor.
        :param n_step: Will generate {n_step}-step-ahead predictions. Defaults to (and cannot be less than) 1.
        :param progress: Should progress-bar be displayed?
        :param initial_prediction: Usually left `None` so that initial predictions are made automatically; in some
        cases case you might pass a StateBelief generated from a previous prediction.
        :param kwargs: Other kwargs that will be passed to the kf's `design.for_batch()` method, which in turn passes
        them to each process (or to the NNs specified in `*_var_predict`). Sometimes, processes might share keyword-
        argument names but you want to pass different arguments to them -- for example, if you have two processes that
        take `predictors` but you want to pass a different predictor matrix to each. In this case you can disambiguate
        using sklearn-style double-underscoring: `process1__predictors` and `process2__predictors`.
        :return: A StateBeliefOverTime consisting of n-step-ahead predictions.
        """
        if not args:
            if initial_prediction is None:
                raise ValueError("No input `args` were passed, so must pass `initial_prediction`.")
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
        assert out_timesteps > 0

        assert n_step > 0

        progress = progress or identity
        if progress is True:
            progress = tqdm
        times = progress(range(1, out_timesteps))

        try:
            design_for_batch = self.design.for_batch(num_groups=num_groups, num_timesteps=out_timesteps + n_step - 1, **kwargs)
        except IndexError as e:
            if (n_step > 1 or forecast_horizon > 0) and ("out of bounds for dimension" in str(e)):
                raise ValueError(
                    f"Hit an index error when setting up design. If you passed external predictors, make sure they "
                    f"extend into the future to support `n_step`/`forecast_horizon`; or reduce `out_timesteps` "
                    f"(currently {out_timesteps:,})."
                ) from e
            else:
                raise e

        # initial state of the system:
        if initial_prediction is None:
            state_pred_1step = self._predict_initial_state(design_for_batch)
        else:
            state_pred_1step = initial_prediction.copy()
        state_pred_1step.compute_measurement(H=design_for_batch.H(0), R=design_for_batch.R(0))

        state_preds = [state_pred_1step]
        for t1 in times:
            t = t1 - 1

            # reconcile last timestep's 1step prediction with what was actually measured:
            if args:
                state_pred = state_pred_1step.update(*args, time=t)
            else:
                state_pred = state_pred_1step.copy()

            # predict
            # F/Q at t is transition *from* t *to* t+1
            for i in range(n_step):
                state_pred = state_pred.predict(F=design_for_batch.F(t + i), Q=design_for_batch.Q(t + i))
                if i == 0:
                    # always need to save the 1step for the next iter, even if it's not the output:
                    state_pred_1step = state_pred
                    state_pred_1step.compute_measurement(H=design_for_batch.H(t1), R=design_for_batch.R(t1))

            # if not 1step, need to additionally compute measurement for output:
            if i > 0:
                state_pred.compute_measurement(H=design_for_batch.H(t1 + i), R=design_for_batch.R(t1 + i))
            state_preds.append(state_pred)

        return self.family.concatenate_over_time(state_beliefs=state_preds, design=self.design)

    def _predict_initial_state(self, design_for_batch: Design) -> 'Gaussian':
        return self.family(
            means=design_for_batch.initial_mean,
            covs=design_for_batch.initial_covariance,
            # we consider this a one-step-ahead prediction, so last measured one step ago:
            last_measured=torch.ones(design_for_batch.num_groups, dtype=torch.int)
        )
