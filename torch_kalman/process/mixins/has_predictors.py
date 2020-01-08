import torch


class HasPredictors:

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  predictors: torch.Tensor,
                  expected_num_predictors: int,
                  allow_extra_timesteps: bool = True):

        if not isinstance(predictors, torch.Tensor):
            raise ValueError(f"Process {self.id} received 'predictors' that is not a Tensor.")
        elif predictors.requires_grad:
            raise ValueError(f"Process {self.id} received 'predictors' that requires_grad, which is not allowed.")
        elif torch.isnan(predictors).any():
            raise ValueError(f"Process {self.id} received 'predictors' that has nans.")

        mm_num_groups, mm_num_ts, mm_num_preds = predictors.shape
        if mm_num_groups != num_groups:
            raise ValueError(f"Batch-size is {num_groups}, but predictors.shape[0] is {mm_num_groups}.")
        if mm_num_ts != num_timesteps:
            if (not allow_extra_timesteps) or (mm_num_ts < num_timesteps):
                msg = f"Batch num. timesteps is {num_timesteps}, but predictors.shape[1] is {mm_num_ts}."
                if mm_num_ts < num_timesteps:
                    msg += (f" This can happen if `forecast_horizon` is longer than the predictors; try reducing by "
                            f"{num_timesteps - mm_num_ts}")
                raise ValueError(msg)
        if mm_num_preds != expected_num_predictors:
            raise ValueError(f"`predictors.shape[2]` = {mm_num_preds}, but expected {expected_num_predictors}")

        return super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps)
