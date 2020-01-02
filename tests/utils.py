from torch_kalman.design import Design
from torch_kalman.process import LocalTrend


def simple_mv_velocity_design(dims: int = 2):
    processes, measures = [], []
    for i in range(dims):
        process = LocalTrend(id=str(i), decay_velocity=False)
        measure = str(i)
        process.add_measure(measure=measure)
        processes.append(process)
        measures.append(measure)
    return Design(processes=processes, measures=measures)
