from typing import Dict, Tuple

from torch import Tensor
from torch_kalman.internals.repr import NiceRepr
from torch_kalman.process.utils.design_matrix.utils import SeqOfTensors


class DynamicMatrix(NiceRepr):
    _repr_attrs = ()

    def __init__(self, base_mat: Tensor, dynamic_assignments: Dict[Tuple[int, int], SeqOfTensors]):
        self.base_mat = base_mat
        self.dynamic_assignments = dynamic_assignments

    def __call__(self, t: int) -> Tensor:
        out = self.base_mat
        if self.dynamic_assignments:
            out = out.clone()
        for (r, c), values in self.dynamic_assignments.items():
            out[..., r, c] = values[t]
        return out
