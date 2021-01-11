from math import pi
from typing import Union
import numpy as np
import torch

from torch_kalman.utils.datetime import DateTimeHelper


def fourier_model_mat(datetimes: np.ndarray,
                      K: int,
                      period: Union[np.timedelta64, str],
                      output_fmt: str = 'float64') -> np.ndarray:
    """
    :param datetimes: An array of datetimes.
    :param K: The expansion integer.
    :param period: Either a np.timedelta64, or one of {'weekly','yearly','daily'}
    :param start_datetime: A np.datetime64 on which to consider the season-start; useful for aligning (e.g) weekly
    seasons to start on Monday, or daily seasons to start on a particular hour. Default is first monday after epoch.
    :param output_fmt: A numpy dtype, or 'dataframe' to output a dataframe.
    :return: A numpy array (or dataframe) with the expanded fourier series.
    """
    # parse period:
    name = 'fourier'
    if isinstance(period, str):
        name = period
        if period == 'weekly':
            period = np.timedelta64(7, 'D')
        elif period == 'yearly':
            period = np.timedelta64(int(365.25 * 24), 'h')
        elif period == 'daily':
            period = np.timedelta64(24, 'h')
        else:
            raise ValueError("Unrecognized `period`.")

    period_int = period.view('int64')
    dt_helper = DateTimeHelper(dt_unit=np.datetime_data(period)[0])
    time = dt_helper.validate_datetimes(datetimes).view('int64')

    output_dataframe = (output_fmt.lower() == 'dataframe')
    if output_dataframe:
        output_fmt = 'float64'

    # fourier matrix:
    out_shape = tuple(datetimes.shape) + (K * 2,)
    out = np.empty(out_shape, dtype=output_fmt)
    columns = []
    for idx in range(K):
        k = idx + 1
        for is_cos in range(2):
            val = 2. * np.pi * k * time / period_int
            out[..., idx * 2 + is_cos] = np.sin(val) if is_cos == 0 else np.cos(val)
            columns.append(f"{name}_K{k}_{'cos' if is_cos else 'sin'}")

    if output_dataframe:
        if len(out_shape) > 2:
            raise ValueError("Cannot output dataframe when input is 2+D array.")
        from pandas import DataFrame
        out = DataFrame(out, columns=columns)

    return out


def fourier_tensor(time: torch.Tensor, seasonal_period: float, K: int) -> torch.Tensor:
    """
    Given an N-dimensional tensor, create an N+2 dimensional tensor with the 2nd to last dimension corresponding to the
    Ks and the last dimension corresponding to sin/cos.
    """
    out = torch.empty((*time.shape, K, 2))
    base_index = tuple(slice(0, x) for x in time.shape)
    for idx in range(K):
        k = idx + 1
        for sincos in range(2):
            val = 2. * pi * k * time / seasonal_period
            index = base_index + (idx, sincos)
            out[index] = torch.sin(val) if sincos == 0 else torch.cos(val)
    return out
