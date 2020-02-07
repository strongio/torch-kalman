from typing import Sequence, Tuple, Dict, Optional, Callable

import torch
from torch_kalman.process.utils.design_matrix.base import DesignMatrix, DesignMatAssignment


class TransitionMatrix(DesignMatrix):
    dim1_name = 'to_element'
    dim2_name = 'from_element'

    @property
    def from_elements(self):
        return self.dim1_names

    @property
    def to_elements(self):
        return self.dim2_names


class MeasureMatrix(DesignMatrix):
    dim1_name = 'measure'
    dim2_name = 'state_element'

    @property
    def measures(self):
        return self.dim1_names

    @property
    def state_elements(self):
        return self.dim2_names


class VarianceMultiplierMatrix(DesignMatrix):
    """
    A diagonal-only matrix that can be multiplied by a covariance matrix.
    """

    def __init__(self, elements: Sequence[str], nonzero_elements: Optional[Sequence[str]] = None):
        if nonzero_elements is None:
            nonzero_elements = elements
        super().__init__(dim1_names=elements, dim2_names=elements)
        for element in nonzero_elements:
            self.assign(**{self.dim1_name: element}, value=0.0)
            self.set_ilink(**{self.dim1_name: element}, ilink=torch.exp)

    @classmethod
    def _from_attributes(cls,
                         dim1_names: Sequence,
                         dim2_names: Sequence,
                         batch_info: Tuple,
                         new_assignments: Dict,
                         new_ilinks: Dict
                         ) -> 'VarianceMultiplierMatrix':
        """
        Ignores nonzero_elements, but that gets overwritten by new_assignments and new_ilinks
        """
        assert dim1_names == dim2_names
        out = cls(elements=dim1_names)
        out._batch_info = batch_info
        out._assignments = new_assignments
        out._ilinks = new_ilinks
        return out

    def assign(self, value: DesignMatAssignment, **kwargs):
        if value != 0.0:
            raise ValueError(f"Cannot override assignment-value for {type(self).__name__}.")
        super().assign(value=value, **kwargs)

    def set_ilink(self, ilink: Optional[Callable], **kwargs):
        if ilink is not torch.exp:
            raise ValueError(f"Cannot override ilink for {type(self).__name__}.")
        super().set_ilink(ilink=ilink, **kwargs)


class ProcessVarianceMultiplierMatrix(VarianceMultiplierMatrix):
    dim1_name = 'state_element'
    dim2_name = 'state_element'

    @property
    def state_elements(self):
        return self.dim1_names


class MeasureVarianceMultiplierMatrix(VarianceMultiplierMatrix):
    dim1_name = 'measure'
    dim2_name = 'measure'

    @property
    def measures(self):
        return self.dim1_names
