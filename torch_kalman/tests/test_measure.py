from torch_kalman.measure import Measure, MeasureForBatch
from torch_kalman.process.velocity import Velocity
from torch_kalman.tests import TestCaseTK


class TestMeasure(TestCaseTK):
    def test_measure_attrs(self):
        measure = Measure(id='test')

        measure.add_process(process=Velocity(id='test'), value=1.0)

        self.assertEqual(len(measure.processes), 1)

        batch_measure = measure.for_batch(batch_size=2)
        self.assertIsInstance(batch_measure, MeasureForBatch)
        self.assertEqual(batch_measure.batch_size, 2)
