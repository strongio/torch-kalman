import math
from typing import Union
import numpy as np
import torch


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

    if not isinstance(datetimes, np.ndarray) and isinstance(getattr(datetimes, 'values', None), np.ndarray):
        datetimes = datetimes.values
    period_int = int(period / np.timedelta64(1, 'ns'))
    time_int = (datetimes.astype("datetime64[ns]") - np.datetime64(0, 'ns')).astype('int64')

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
            val = 2. * np.pi * k * time_int / period_int
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
    out = torch.empty(time.shape + (K, 2))
    for idx in range(K):
        k = idx + 1
        for sincos in range(2):
            val = 2. * math.pi * k * time / seasonal_period
            out[..., idx, sincos] = torch.sin(val) if sincos == 0 else torch.cos(val)
    return out
