import unittest

from torch_kalman.batch import Batchable


class TestBatchable(unittest.TestCase):
    def test_batchable(self):
        instance = Batchable()

        with self.assertRaises(ValueError):
            instance.batch_info = (0, 1)

        instance.batch_info = (3, 1)

        with self.assertRaises(RuntimeError):
            instance.batch_info = (4, 1)
