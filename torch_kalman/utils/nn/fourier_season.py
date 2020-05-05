from typing import Union

import torch
import numpy as np
from torch.nn import Linear
from torch_kalman.utils.datetime import DateTimeHelper
from torch_kalman.utils.features import fourier_model_mat


class FourierSeasonNN(Linear):
    def __init__(self,
                 K: int,
                 period: Union[np.timedelta64, str],
                 dt_unit: str,
                 num_outputs: int,
                 bias: bool = False):
        self.K = K
        self.period = period
        self._dt_helper = DateTimeHelper(dt_unit=dt_unit)
        super().__init__(in_features=K * 2, out_features=num_outputs, bias=bias)

    def forward(self, datetimes: np.ndarray) -> torch.Tensor:
        return super().forward(self._datetimes_to_tensor(datetimes))

    def _datetimes_to_tensor(self, datetimes: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(fourier_model_mat(
            datetimes=datetimes,
            K=self.K,
            period=self.period,
            output_fmt='float32'
        ))

    def reset_parameters(self):
        super().reset_parameters()
        self.weight.data *= .10
