from typing import Union, Sequence, Dict

from numpy.core.multiarray import ndarray
from torch import Tensor

import numpy as np

# assumed by `day_of_week_num`:
assert np.zeros(1).astype('datetime64[D]') == np.datetime64('1970-01-01', 'D')


def day_of_week_num(dts: np.ndarray) -> np.ndarray:
    return (dts.astype('datetime64[D]').view('int64') - 4) % 7


def tens_to_long(tens: Union[ndarray, Tensor], **kwargs) -> Sequence[Dict[str, Union[float, int]]]:
    """
    Convert a multidimensional array into "long" format. Useful because it generalizes to tensors of any dimensionality,
     but be warned this is not particularly fast or memory-efficient.

    :param tens: A multidimensional array, e.g. a numpy ndarray, pytorch Tensor, etc.
    :return: A list of dictionaries, with keys dim0-N specifying the index, and a 'value' key specifying the value at
      that index.
    """
    shape = tens.shape
    dims = len(shape)

    row = kwargs.get('row', {})
    dim_num = kwargs.get('dim_num', 0)

    rows = list()
    for i in range(shape[0]):
        new_row = row.copy()
        new_row['dim{}'.format(dim_num)] = i
        if dims > 1:
            rows.extend(tens_to_long(tens[i], row=new_row, dim_num=dim_num + 1))
        else:
            new_row['value'] = tens[i].item()
            rows.append(new_row)

    return rows
