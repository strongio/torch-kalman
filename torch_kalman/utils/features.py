from typing import Union, Optional
import numpy as np


def fourier_model_mat(dt: np.ndarray,
                      K: int,
                      period: Union[np.timedelta64, str],
                      start_dt: Optional[np.datetime64] = None,
                      output_dataframe: bool = False) -> np.ndarray:
    """
    :param dt: An array of datetimes.
    :param K: The expansion integer.
    :param period: Either a np.timedelta64, or one of {'weekly','yearly','daily'}
    :param start_dt: A np.datetime64 on which to consider the season-start; useful for aligning (e.g) weekly seasons to
    start on Monday, or daily seasons to start on a particular hour. If None is passed will use the linux epoch, as
    numpy does.
    :param output_dataframe: If True, output a pandas dataframe; if False (the default) a numpy array.
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
    if hasattr(dt, 'values'):
        dt = dt.values
    time_unit = np.datetime_data(period)[0]
    if start_dt is None:
        time = dt.astype(f'datetime64[{time_unit}]').view('int64')
    else:
        if hasattr(dt, 'to_datetime64'):
            dt = dt.to_datetime64()
        time = (dt - start_dt).astype(f'timedelta64[{time_unit}]').view('int64')
    period_int = period.view('int64')

    # fourier matrix:
    out = np.empty((len(dt), K * 2))
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
