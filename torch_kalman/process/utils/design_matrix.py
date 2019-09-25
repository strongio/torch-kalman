import functools
from typing import Optional, Sequence, Union, Callable, Tuple, List, Hashable, Dict

import torch
from torch import Tensor

from torch_kalman.utils import bifurcate

DesignMatAssignment = Union[float, Tensor, Callable]
SeqOfTensors = Union[Tuple[Tensor], List[Tensor]]
DesignMatAdjustment = Union[Tensor, SeqOfTensors]


def _is_dynamic_assignment(x) -> bool:
    return isinstance(x, (list, tuple))


class is_for_batch:
    def __init__(self, setting: bool):
        self.setting = setting

    def __call__(self, func: Callable):
        msg = "Can only" if self.setting else "Cannot"

        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            if not self._for_batch:
                raise ValueError(f"{msg} adjust a f{self.__class__.__name__} if it's the output of `for_batch()`.")
            return func(self, *args, **kwargs)

        return wrapped


# class DynamicDesignMatrix:
#     def __init__(self,
#                  shape: Tuple,
#                  base_assignments: Dict[Tuple, Tensor],
#                  dynamic_assignments: Dict[Tuple, SeqOfTensors],
#                  ilinks: Dict[Tuple, Callable]):
#         self.shape = shape
#         self.base_assignments = base_assignments
#         self.dynamic_assignments = dynamic_assignments
#         self.ilinks = ilinks
#
#     def __call__(self, t: int):
#         mat = torch.zeros(self.shape)
#         for rc, value in self.base_assignments.items():
#             mat[rc] = value


class DesignMatrix:
    dim1_name: str = None
    dim1_fixed: bool = None
    dim2_name: str = None
    dim2_fixed: bool = None
    default_ilink = None

    def __init_subclass__(cls, **kwargs):
        for attr in ('dim1_name', 'dim1_fixed', 'dim2_name', 'dim2_fixed'):
            if attr is None:
                raise TypeError(f"{cls.__name__} must set `{attr}` attribute.")
        super().__init_subclass__(**kwargs)

    def __init__(self,
                 process_id: str,
                 dim1_names: Optional[Sequence[Hashable]] = None,
                 dim2_names: Optional[Sequence[Hashable]] = None):
        self.process_id = process_id
        self.dim1_names = [] if dim1_names is None else list(dim1_names)
        self.dim2_names = [] if dim2_names is None else list(dim2_names)
        self._assignments = {}
        self._ilinks = {}
        self._for_batch = False

    # base ------------------------------------
    @property
    def empty(self) -> bool:
        return not self._assignments

    @is_for_batch(False)
    def assign(self, value: DesignMatAssignment, **kwargs):
        key = self._get_key(kwargs)
        self._check_or_expand_dims(*key)
        value = self._validate_assignment(value)
        if key in self._assignments:
            raise ValueError(f"Already have assignment for {key}.")
        self._assignments[key] = value

    @staticmethod
    def _validate_assignment(value: DesignMatAssignment) -> Union[Tensor, Callable]:
        if isinstance(value, float):
            value = torch.Tensor([value])
        elif isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError("Design-mat assignment should have only one-element.")
            if value.grad_fn:
                raise ValueError("If `value` is the result of computations that require_grad, need to wrap those "
                                 "computations in a function and pass that for `value` instead.")
        elif not callable(value):
            raise ValueError("`value` must be float, tensor, or a function that produces a tensor")
        return value

    @is_for_batch(False)
    def set_ilink(self, ilink: Optional[Callable], **kwargs):
        key = self._get_key(kwargs)
        if key not in self._assignments:
            raise ValueError(f"Tried to set ilink for {key} but must `assign` first.")
        if ilink is None:
            ilink = self.default_ilink
        assert ilink is None or callable(ilink)
        self._ilinks[key] = ilink

    # for batch -----------------------
    @is_for_batch(False)
    def for_batch(self) -> 'DesignMatrix':
        # shallow-copy dimnames:
        out = self.__class__(dim1_names=list(self.dim1_names), dim2_names=list(self.dim2_names))
        # convert assignments to lists so adjustments can be appended:
        out._assignments = {key: [value() if callable(value) else value] for key, value in self._assignments.items()}
        # shallow-copy ilinks:
        out._ilinks = dict(self._ilinks)
        # disable setting assignments/ilinks, enable adjustments:
        out._for_batch = True
        return out

    @is_for_batch(True)
    def adjust(self, value: DesignMatAdjustment, **kwargs):
        key = self._get_key(kwargs)
        value = self._validate_adjustment(value)
        self._assignments[key].append(value)

    @staticmethod
    def _validate_adjustment(value: DesignMatAdjustment) -> Union[Tensor, SeqOfTensors]:
        raise RuntimeError("TODO(@jwdink)")

    @classmethod
    def merge(cls, mats: Sequence['DesignMatrix']) -> 'DesignMatrix':
        for i, mat in enumerate(mats):
            if not isinstance(mat, cls):
                raise TypeError(f"{mat} must be instance of {cls.__name__}")
            if not mat._for_batch:
                raise ValueError(f"The {i}th input is not valid, is not the result of `for_batch()`.")

        new_dimnames = {}
        for i in (1, 2):
            new_dimnames[i] = []
            if getattr(cls, f"dim{i}_fixed"):
                for mat in mats:
                    new_dimnames[i].extend((mat.process_id, x) for x in getattr(mat, f"dim{i}_names"))
            else:
                for mat in mats:
                    new_dimnames[i].extend(getattr(mat, f"dim{i}_names"))
                # unfixed dims are merged, not concat'd
                new_dimnames[i] = sorted(set(new_dimnames[i]))

        out = cls(process_id='merged', dim1_names=new_dimnames[1], dim2_names=new_dimnames[2])
        out._for_batch = True
        for mat in mats:
            out._assignments.update({(mat.process_id, k): v for k, v in mat._assignments.items()})
            out._ilinks.update({(mat.process_id, k): v for k, v in mat._ilinks.items()})
        return out

    @is_for_batch(True)
    def compile(self, num_groups: int, num_timesteps: int) -> 'DynamicMatrix':
        """
        Consolidate assignments then apply the link function. Some assignments can be "frozen" into a pre-computed
        matrix, while others must remain as lists to be evaluated in DesignForBatch as needed.
        """
        base_mat = torch.zeros(num_groups, len(self.dim1_names), len(self.dim2_names))
        dynamic_assignments = {}
        for (dim1, dim2), values in self._assignments.items():
            r = self.dim1_names.index(dim1)
            c = self.dim2_names.index(dim2)
            ilink = self._ilinks.get((dim1, dim2), self.default_ilink)
            if ilink is None:
                # if using identity link, can take shortcut
                dynamic, base = bifurcate(values, _is_dynamic_assignment)
                base_mat[:, r, c] = torch.sum(base)
                per_timestep = list(zip(*dynamic))
            else:
                # otherwise, need to replicate those static assignments to match each dynamic
                values = [x if _is_dynamic_assignment(x) else [x] * num_timesteps for x in values]
                per_timestep = list(zip(*values))
            assert len(per_timestep) == num_timesteps  # TODO: don't use assert
            dynamic_assignments[(r, c)] = [torch.sum(x) for x in per_timestep]
        return DynamicMatrix(base_mat, dynamic_assignments)

    # utils ------------------------------------------
    def _get_key(self, kwargs: dict) -> Tuple[str, str]:
        extra = set(kwargs).difference({self.dim1_name, self.dim2_name})
        if extra:
            raise TypeError(f"Unexpected argument(s): {extra}.")
        return kwargs.get(self.dim1_name), kwargs.get(self.dim2_name)

    def _check_or_expand_dims(self, dim1, dim2):
        if dim1 not in self.dim1_names:
            if self.dim1_fixed:
                raise ValueError(f"{dim1}" not in {self.dim1_names})
            else:
                self.dim1_names.append(dim1)  # TODO: sorted?
        if dim2 not in self.dim2_names:
            if self.dim2_fixed:
                raise ValueError(f"{dim2}" not in {self.dim2_names})
            else:
                self.dim2_names.append(dim2)


class TransitionMatrix(DesignMatrix):
    # TODO: order?
    dim1_name = 'from_element'
    dim1_fixed = True
    dim2_name = 'to_element'
    dim2_fixed = True

    @property
    def from_elements(self):
        return self.dim1_names

    @property
    def to_elements(self):
        return self.dim1_names


class MeasureMatrix(DesignMatrix):
    dim1_name = 'measure'
    dim1_fixed = False
    dim2_name = 'state_element'
    dim2_fixed = True

    @property
    def measures(self):
        return self.dim1_names


class VarianceMultiplierMatrix(DesignMatrix):
    """
    A diagonal-only matrix that will be multiplied by the process-variance.
    """
    dim1_name = 'state_element'
    dim1_fixed = True
    dim2_name = 'state_element'
    dim2_fixed = True
    default_ilink = torch.exp

    def __init__(self, process_id: str, dim_names: Sequence[str]):
        super().__init__(process_id=process_id, dim1_names=dim_names, dim2_names=dim_names)
        for se in self.state_elements:
            self.assign(state_element=se, value=0.0)
            self.set_ilink(state_element=se, ilink=None)

    def assign(self, value: DesignMatAssignment, **kwargs):
        if value != 0.0:
            raise ValueError(f"Cannot override assignment-value for {self.__class__.__name__}.")
        super().assign(value=value, **kwargs)

    def set_ilink(self, ilink: Optional[Callable], **kwargs):
        if ilink is not None:
            raise ValueError(f"Cannot override ilink for {self.__class__.__name__}.")
        super().set_ilink(ilink=ilink, **kwargs)

    @property
    def state_elements(self):
        return self.dim1_names


class DynamicMatrix:
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
