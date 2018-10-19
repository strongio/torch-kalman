from typing import Union, Sequence, Dict

import torch
from numpy.core.multiarray import ndarray
from torch import Tensor

from math import pi


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


def fourier_series(time: Tensor, seasonal_period: int, parameters: Tensor):
    dim1, dim2 = parameters.shape
    assert dim2 == 2, f"Expected K X 2 matrix, got {(dim1, dim2)}."

    out = torch.zeros_like(time)
    for idx in range(dim1):
        k = idx + 1
        out += (parameters[idx, 0] * torch.sin(2. * pi * k * time / seasonal_period))
        out += (parameters[idx, 1] * torch.cos(2. * pi * k * time / seasonal_period))

    return out
