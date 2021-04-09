import unittest
import numpy as np
import torch
from torchcast.utils import TimeSeriesDataset


class TestUtils(unittest.TestCase):
    def test_pad_x(self, num_times: int = 10):
        from pandas import DataFrame
        df = DataFrame({'x1': np.random.randn(num_times), 'x2': np.random.randn(num_times)})
        df['y'] = 1.5 * df['x1'] + -.5 * df['x2'] + .1 * np.random.randn(num_times)
        df['time'] = df.index.values
        df['group'] = '1'
        dataset1 = TimeSeriesDataset.from_dataframe(
            dataframe=df,
            group_colname='group',
            time_colname='time',
            dt_unit=None,
            X_colnames=['x1', 'x2'],
            y_colnames=['y']
        )
        dataset2 = TimeSeriesDataset.from_dataframe(
            dataframe=df,
            group_colname='group',
            time_colname='time',
            dt_unit=None,
            X_colnames=['x1', 'x2'],
            y_colnames=['y'],
            pad_X=None
        )
        self.assertFalse(torch.isnan(dataset1.tensors[1]).any())
        self.assertFalse(torch.isnan(dataset2.tensors[1]).any())
        self.assertTrue((dataset1.tensors[1] == dataset2.tensors[1]).all())
