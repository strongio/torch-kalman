from typing import Dict, Union

from torch import Tensor

from torch_kalman.process import Process


class Measure:
    def __init__(self, id: str):
        self.id = str(id)
        self.processes = {}

    def add_process(self, process: Process, value: Union[float, Tensor, None]) -> None:
        """
        :param process: A Process.
        :param value: The value that's multiplied by the Process's measurable state to yield (its contribution to) the
        Measure. Set to None to specify that this value needs to be specified on a per-batch basis.
        """
        if process.id in self.processes.keys():
            raise ValueError(f"Process {process.id} is already in this measure.")
        self.processes[process.id] = value

    def for_batch(self, batch_size: int) -> 'MeasureForBatch':
        return MeasureForBatch(measure=self, batch_size=batch_size)


class MeasureForBatch:
    def __init__(self, measure: Measure, batch_size: int):
        self.batch_size = batch_size
        self.measure = measure
        self.batch_processes = {}

    def processes(self):
        return {**self.measure.processes, **self.batch_processes}

    def add_process(self, process: Process, values: Tensor) -> None:
        assert len(values) == 1 or len(values) == self.batch_size

        if self.measure.processes.get(process.id, None):
            raise ValueError(f"The process '{process.id}' was already set for this Measure, so can't give it batch-specific "
                             f"values.")

        if process.id in self.batch_processes.keys():
            raise ValueError(f"The process '{process.id}' was already set for this batch, so can't set it again.")

        self.batch_processes[process.id] = values
