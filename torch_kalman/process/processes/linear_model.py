from typing import Sequence, Optional, Union, Callable, Collection

import torch

from torch import Tensor

from torch_kalman.process import Process
from torch_kalman.internals.utils import split_flat


class LinearModel(Process):
    """
    A process that learns how external predictors map onto measurements.

    In the most common use case, we want the coefficients to be learned for each time-series separately, but with an
    initial estimate (with mean and covariance) that's shared across time-series. As more observations come in, the
    coefficient-estimates for a particular series will converge.

    If process_variance = True, then the coefficient estimates won't converge, but will be allowed to drift over time.

    If init_variance = False, then all time-serieses will be fixed at the same initial estimates, with no uncertainty.
    """

    def __init__(self,
                 id: str,
                 covariates: Sequence[str],
                 init_variance: Union[bool, Collection[str]] = True,
                 process_variance: Union[bool, Collection[str]] = False,
                 inv_link: Optional[Callable] = None):
        """

        :param id: A unique name for the process.
        :param covariates: The names of the predictors.
        :param init_variance: If True (the default), then there is initial uncertainty about the values of the
        coefficients. Can also specify which covariates have uncertainty.
        :param process_variance: If False (the default), then the uncertainty about the values of the coefficients does
        not grow at each timestep, so over time these coefficients eventually converge to a certain value. If True, then
         the coefficients are allowed to 'drift' over time.
        :param inv_link: An inverse link function that maps the linear-model to the prediction; default the identity
        link.
        """
        if isinstance(covariates, str):
            raise TypeError("`covariates` should be sequence of strings, not single string")

        self.inv_link = inv_link

        # process covariance:
        self._dynamic_state_elements = []
        if process_variance:
            self._dynamic_state_elements = covariates if isinstance(process_variance, bool) else process_variance
            extras = set(self._dynamic_state_elements) - set(covariates)
            if len(extras):
                raise ValueError(f"`process_variance` includes items not in `covariates`:{extras}")

        # initial covariance:
        self._fixed_state_elements = []
        if init_variance is False:
            init_variance = []
        if init_variance is not True:
            extras = set(init_variance) - set(covariates)
            if len(extras):
                raise ValueError(f"`init_variance` includes items not in `covariates`:{extras}")
            self._fixed_state_elements = [cov for cov in covariates if cov not in init_variance]

        super().__init__(id=id, state_elements=covariates)

        for cov in covariates:
            self._set_transition(from_element=cov, to_element=cov, value=1.0)

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self._dynamic_state_elements

    @property
    def fixed_state_elements(self):
        return self._fixed_state_elements

    def param_dict(self) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict()  # no parameters

    def add_measure(self, measure: str) -> 'LinearModel':
        for cov in self.state_elements:
            self._set_measure(measure=measure, state_element=cov, value=0., ilink=self.inv_link)
        return self

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  predictors: Tensor,
                  allow_extra_timesteps: bool = True) -> 'LinearModel':
        for_batch = super().for_batch(
            num_groups=num_groups,
            num_timesteps=num_timesteps
        )
        self._validate_predictor_mat(
            num_groups=num_groups,
            num_timesteps=num_timesteps,
            predictor_mat=predictors,
            expected_num_predictors=len(self.state_elements),
            allow_extra_timesteps=allow_extra_timesteps
        )

        if predictors.shape[1] > num_timesteps:
            predictors = predictors[:, 0:num_timesteps, :]

        for measure in self.measures:
            for i, cov in enumerate(self.state_elements):
                for_batch._adjust_measure(
                    measure=measure,
                    state_element=cov,
                    adjustment=split_flat(predictors[:, :, i], dim=1)
                )

        return for_batch

    def _validate_predictor_mat(self,
                                num_groups: int,
                                num_timesteps: int,
                                predictor_mat: torch.Tensor,
                                expected_num_predictors: int,
                                allow_extra_timesteps: bool = True):

        if not isinstance(predictor_mat, torch.Tensor):
            raise ValueError(f"Process {self.id} received 'predictor_mat' that is not a Tensor.")
        elif predictor_mat.requires_grad:
            raise ValueError(f"Process {self.id} received 'predictor_mat' that requires_grad, which is not allowed.")
        elif torch.isnan(predictor_mat).any():
            raise ValueError(f"Process {self.id} received 'predictor_mat' that has nans.")

        if len(predictor_mat.shape) == 2:
            mm_num_groups, mm_num_preds = predictor_mat.shape
            mm_num_ts = None
        else:
            mm_num_groups, mm_num_ts, mm_num_preds = predictor_mat.shape

        if mm_num_groups != num_groups:
            raise ValueError(f"Batch-size is {num_groups}, but predictor_mat.shape[0] is {mm_num_groups}.")
        if (mm_num_ts is not None) and (mm_num_ts != num_timesteps):
            if (not allow_extra_timesteps) or (mm_num_ts < num_timesteps):
                msg = f"Batch num. timesteps is {num_timesteps}, but predictor_mat.shape[1] is {mm_num_ts}."
                if mm_num_ts < num_timesteps:
                    msg += (f" This can happen if `forecast_horizon` is longer than the predictors; try reducing by "
                            f"{num_timesteps - mm_num_ts}")
                raise ValueError(msg)
        if mm_num_preds != expected_num_predictors:
            raise ValueError(f"`predictor_mat.shape[2]` = {mm_num_preds}, but expected {expected_num_predictors}")
