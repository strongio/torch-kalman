import numpy as np
import torch

from torch_kalman.data_utils import TimeSeriesBatch

from torch_kalman.state_belief.over_time import StateBeliefOverTime

try:
    from pandas import DataFrame, concat
except ImportError:
    DataFrame = None
    concat = None


def get_prediction_df(state_beliefs: StateBeliefOverTime,
                      batch: TimeSeriesBatch,
                      group_colname: str,
                      time_colname: str) -> DataFrame:
    """
    Get a dataframe of predictions + their uncertainty.

    :param state_beliefs: A StateBeliefOverTime (output of KalmanFilter)
    :param batch: The TimeSeriesBatch that was input to the KalmanFilter
    :param group_colname: Column-name for 'group'
    :param time_colname: Column-name for 'time'
    :return: A pandas DataFrame with group, time, measure, predicted_mean, predicted_std
    """
    num_timesteps = len(state_beliefs.state_beliefs)
    if batch.tensor.shape[1] > num_timesteps:
        raise ValueError("The `batch` has more timesteps than the `state_beliefs`")

    times = batch.start_times[:, None] + np.arange(0, num_timesteps)

    stds = torch.diagonal(state_beliefs.prediction_uncertainty, dim1=-1, dim2=-2).sqrt()
    out = []
    for i, measure in enumerate(batch.measures):
        # mean:
        df_pred = TimeSeriesBatch.tensor_to_dataframe(
            tensor=state_beliefs.predictions[..., [i]],
            times=times,
            group_names=batch.group_names,
            measures=['predicted_mean'],
            group_colname=group_colname,
            time_colname=time_colname
        )

        # std:
        df_std = TimeSeriesBatch.tensor_to_dataframe(
            tensor=stds[..., [i]],
            times=times,
            group_names=batch.group_names,
            measures=['predicted_std'],
            group_colname=group_colname,
            time_colname=time_colname
        )

        out.append(df_pred.merge(df_std, on=[group_colname, time_colname]))
        out[-1]['measure'] = measure

    return concat(out)


def get_components_df(state_beliefs: StateBeliefOverTime,
                      batch: TimeSeriesBatch,
                      group_colname: str,
                      time_colname: str) -> DataFrame:
    """
    Get a dataframe describing the components of a StateBeliefOverTime.

    :param state_beliefs: A StateBeliefOverTime (output of KalmanFilter)
    :param batch: The TimeSeriesBatch that was input to the KalmanFilter
    :param group_colname: Column-name for 'group'
    :param time_colname: Column-name for 'time'
    :return: A pandas DataFrame with group, time, process, state_element, measure, value, std
    """

    if DataFrame is None:
        raise ImportError("Requires `pandas` package to be installed.")

    num_timesteps = len(state_beliefs.state_beliefs)
    if batch.tensor.shape[1] > num_timesteps:
        raise ValueError("The `batch` has more timesteps than the `state_beliefs`")

    times = batch.start_times[:, None] + np.arange(0, num_timesteps)

    def _tensor_to_df(tens, measures):
        return batch.tensor_to_dataframe(tensor=tens,
                                         times=times,
                                         group_names=batch.group_names,
                                         group_colname=group_colname,
                                         time_colname=time_colname,
                                         measures=measures)

    # components:
    dfs = []
    for (measure, process, state_element), (m, std) in state_beliefs.components().items():
        df = _tensor_to_df(torch.stack([m, std], 2), measures=['value', 'std'])
        df['process'], df['state_element'], df['measure'] = process, state_element, measure
        dfs.append(df)

    # residuals:
    orig_padded = state_beliefs.predictions.data.clone()
    orig_padded[:] = np.nan
    orig_padded[:, 0:batch.tensor.shape[1], :] = batch.tensor
    dfr = _tensor_to_df(state_beliefs.predictions - orig_padded, measures=batch.measures)
    for measure in batch.measures:
        df = dfr.loc[:, [group_colname, time_colname, measure]].copy()
        df['process'], df['state_element'], df['measure'] = 'residuals', 'residuals', measure
        df['value'] = df.pop(measure)
        dfs.append(df)

    return concat(dfs, sort=True)
