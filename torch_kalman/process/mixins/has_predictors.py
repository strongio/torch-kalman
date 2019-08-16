import torch


class HasPredictorsMixin:
    for_batch_kwargs = ['predictors']  # TODO: append, don't replace

    def _check_predictor_tens(self,
                              predictors: torch.Tensor,
                              num_groups: int,
                              num_timesteps: int,
                              num_measures: int,
                              allow_extra_timesteps: bool = False):
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
                f"Batch num. timesteps is {num_timesteps}, but predictors.shape[1] is {mm_num_ts}."
        if mm_num_preds != num_measures:
            raise ValueError(f"`predictors.shape[2]` = {mm_num_preds}, but expected {num_measures}")
