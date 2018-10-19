from typing import Union, Sequence, Dict

from numpy.core.multiarray import ndarray
from torch import Tensor


def tens_to_long(tens: Union[ndarray, Tensor], **kwargs) -> Sequence[Dict[str, Union[float, int]]]:
    """
    Convert a multidimensional array into "long" format.

    :param tens: A multidimensional array, e.g. a numpy ndarray, pytorch Tensor, etc.
    :return: A list of dictionaries, with keys dim0-N specifying the index, and a 'value' key specifying the value at that
     index.
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
