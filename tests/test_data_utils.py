import unittest
from warnings import warn

import torch

from torch_kalman.utils.data import TimeSeriesDataset


class TestDataUtils(unittest.TestCase):
    def test_time_series_dataset(self):
        values = torch.randn((3, 39, 2))

        batch = TimeSeriesDataset(
            values,
            group_names=['one', 'two', 'three'],
            start_times=[0, 0, 0],
            measures=[['y1', 'y2']],
            dt_unit=None
        )
        try:
            import pandas as pd
        except ImportError:
            warn("Not testing TimeSeriesDataset.to_dataframe, pandas not installed.")
            return
        df1 = batch.to_dataframe()

        df2 = pd.concat([
            pd.DataFrame(values[i].numpy(), columns=batch.all_measures).assign(group=group, time=batch.times()[0])
            for i, group in enumerate(batch.group_names)
        ])
        self.assertTrue((df1 == df2).all().all())

    def test_last_measured_idx(self):
        tens = torch.zeros((3, 10, 2))

        # first group 4
        tens[0, 5:, :] = float('nan')

        # second group 7:
        tens[1, 5:, 0] = float('nan')
        tens[1, 8:, 1] = float('nan')

        # third group end:
        tens[2, 8, :] = float('nan')

        d = TimeSeriesDataset(tens, group_names=range(3), start_times=[0] * 3, measures=[['x', 'y']], dt_unit=None)
        last_measured = d._last_measured_idx()[0]

        self.assertEqual(last_measured[0], 4)
        self.assertEqual(last_measured[1], 7)
        self.assertEqual(last_measured[2], 9)
