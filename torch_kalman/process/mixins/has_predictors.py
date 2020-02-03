import torch


class HasPredictors:

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
