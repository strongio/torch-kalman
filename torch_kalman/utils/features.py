from typing import Union, Optional
import numpy as np
from torch_kalman.config import DEFAULT_SEASON_START


def fourier_model_mat(datetimes: np.ndarray,
                      K: int,
                      period: Union[np.timedelta64, str],
                      start_datetime: Optional[np.datetime64] = None,
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

    # convert datetimes and period into ints:
    if hasattr(datetimes, 'values'):
        datetimes = datetimes.values
    time_unit = np.datetime_data(period)[0]
    if hasattr(datetimes, 'to_datetime64'):
        datetimes = datetimes.to_datetime64()
    if start_datetime is None:
        start_datetime = DEFAULT_SEASON_START
    if isinstance(start_datetime, str):
        start_datetime = np.datetime64(start_datetime)
    time = (datetimes - start_datetime).astype(f'timedelta64[{time_unit}]').view('int64')
    period_int = period.view('int64')

    output_dataframe = (output_fmt.lower() == 'dataframe')
    if output_dataframe:
        output_fmt = 'float64'

    # fourier matrix:
    out = np.empty((len(datetimes), K * 2), dtype=output_fmt)
    columns = []
    for idx in range(K):
        k = idx + 1
        for is_cos in range(2):
            val = 2. * np.pi * k * time / period_int
            out[:, idx * 2 + is_cos] = np.sin(val) if is_cos == 0 else np.cos(val)
            columns.append(f"{name}_K{k}_{'cos' if is_cos else 'sin'}")

    if output_dataframe:
        from pandas import DataFrame
        out = DataFrame(out, columns=columns)

    return out
