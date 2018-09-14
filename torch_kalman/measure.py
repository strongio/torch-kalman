from torch import Tensor

from torch_kalman.process import Process


class Measure:
    def __init__(self, id: str):
        self.id = str(id)
        self.processes = {}

    def add_process(self, process: Process, value: (float, Tensor)):
        self.processes[process.id] = value

    def for_batch(self, batch_size: int):
        return MeasureForBatch(measure=self, batch_size=batch_size)


class MeasureForBatch:
    def __init__(self, measure: Measure, batch_size: int):
        self.batch_size = batch_size
        self.processes = measure.processes

    def add_process(self, process: Process, values: Tensor):
        assert len(values) == 1 or len(values) == self.batch_size
        self.processes[process.id] = values
