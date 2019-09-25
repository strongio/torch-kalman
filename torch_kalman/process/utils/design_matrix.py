import itertools
from typing import Optional, Sequence, Union, Callable, Tuple, List, Hashable, Dict

import torch
from torch import Tensor

from torch_kalman.process.utils.for_batch import is_for_batch
from torch_kalman.utils import bifurcate, is_slow_grad

DesignMatAssignment = Union[float, Tensor, Callable]
SeqOfTensors = Union[Tuple[Tensor], List[Tensor]]
DesignMatAdjustment = Union[Tensor, SeqOfTensors]


class DesignMatrix:
    dim1_name: str = None
    dim2_name: str = None

    def __init_subclass__(cls, **kwargs):
        for attr in ('dim1_name', 'dim2_name'):
            if attr is None:
                raise TypeError(f"{cls.__name__} must set `{attr}` attribute.")
        super().__init_subclass__(**kwargs)

    def __init__(self,
                 dim1_names: Optional[Sequence[Hashable]] = None,
                 dim2_names: Optional[Sequence[Hashable]] = None):
        if dim1_names is None and not self._is_measure_dim(1):
            raise ValueError("Must supply dim1_names")
        self.dim1_names = list(dim1_names or [])

        if dim2_names is None and not self._is_measure_dim(2):
            raise ValueError("Must supply dim2_names")
        self.dim2_names = list(dim2_names or [])

        self._assignments = {}
        self._ilinks = {}
        self._for_batch = False

    # base ------------------------------------
    @property
    def empty(self) -> bool:
        return not self._assignments

    @is_for_batch(False)
    def assign(self, value: DesignMatAssignment, force: bool = False, **kwargs):
        key = self._get_key(kwargs)
        self._check_or_expand_dims(*key)
        value = self._validate_assignment(value)
        if key in self._assignments and not force:
            raise ValueError(f"Already have assignment for {key}.")
        self._assignments[key] = value

    @is_for_batch(False)
    def set_ilink(self, ilink: Optional[Callable], force: bool = False, **kwargs):
        key = self._get_key(kwargs)
        if key not in self._assignments:
            raise ValueError(f"Tried to set ilink for {key} but must `assign` first.")
        if key in self._ilinks and not force:
            raise ValueError(f"Already have ilink for {key}.")
        assert ilink is None or callable(ilink)
        self._ilinks[key] = ilink

    # for batch -----------------------
    @property
    @is_for_batch(True)
    def num_groups(self) -> int:
        return self._for_batch[0]

    @property
    @is_for_batch(True)
    def num_timesteps(self) -> int:
        return self._for_batch[1]

    @is_for_batch(False)
    def for_batch(self, num_groups: int, num_timesteps: int) -> 'DesignMatrix':
        # shallow-copy dimnames:
        out = self.__class__(dim1_names=list(self.dim1_names), dim2_names=list(self.dim2_names))
        # convert assignments to lists so adjustments can be appended:
        out._assignments = {key: [value() if callable(value) else value] for key, value in self._assignments.items()}
        # shallow-copy ilinks:
        out._ilinks = dict(self._ilinks)
        # disable setting assignments/ilinks, enable adjustments:
        out._for_batch = (num_groups, num_timesteps)
        return out

    @is_for_batch(True)
    def adjust(self, value: DesignMatAdjustment, check_slow_grad: bool, **kwargs):
        key = self._get_key(kwargs)
        value = self._validate_adjustment(value, check_slow_grad=check_slow_grad)
        self._assignments[key].append(value)

    @classmethod
    def merge(cls, mats: Sequence[Tuple[str, 'DesignMatrix']]) -> 'DesignMatrix':
        if not all(isinstance(mat, cls) for _, mat in mats):
            raise RuntimeError(f"All mats should be instance of {cls.__name__}")
        dims = set(mat._for_batch for _, mat in mats)
        if len(dims) == 1:
            raise RuntimeError(f"All mats should have same num-groups/num-timesteps, got: {dims}")

        # new dimnames:
        new_dimnames = {}
        for i in (1, 2):
            if cls._is_measure_dim(i):
                new_dimnames[i] = set(itertools.chain.from_iterable(getattr(mat, f"dim{i}_names") for _, mat in mats))
            else:
                new_dimnames[i] = []
                for process_name, mat in mats:
                    new_dimnames[i].extend((process_name, x) for x in getattr(mat, f"dim{i}_names"))

        # new assignments/ilinks:
        new_ilinks = {}
        new_assignments = {}
        for process_name, mat in mats:
            new_assignments.update({cls._rename_merged_key(k, process_name): v for k, v in mat._assignments})
            new_ilinks.update({cls._rename_merged_key(k, process_name): v for k, v in mat._ilinks})

        out = cls(dim1_names=list(new_dimnames[1]), dim2_names=list(new_dimnames[2]))
        out._for_batch = list(dims)[0]
        out._assignments = new_assignments
        out._ilinks = new_ilinks
        return out

    @is_for_batch(True)
    def compile(self) -> 'DynamicMatrix':
        """
        Consolidate assignments then apply the link function. Some assignments can be "frozen" into a pre-computed
        matrix, while others must remain as lists to be evaluated in DesignForBatch as needed.
        """
        num_groups, num_timesteps = self._for_batch
        base_mat = torch.zeros(num_groups, len(self.dim1_names), len(self.dim2_names))
        dynamic_assignments = {}
        for (dim1, dim2), values in self._assignments.items():
            r = self.dim1_names.index(dim1)
            c = self.dim2_names.index(dim2)
            ilink = self._ilinks[(dim1, dim2)]
            dynamic, base = bifurcate(values, _is_dynamic_assignment)
            if ilink is None or not dynamic:
                # if using identity link, or there are no dynamic, can take shortcut
                base_mat[:, r, c] = torch.sum(base)
            else:
                # otherwise, need to replicate those static assignments to match each dynamic
                dynamic = dynamic + [[x] * num_timesteps for x in base]
            if dynamic:
                per_timestep = list(zip(*dynamic))
                assert len(per_timestep) == num_timesteps
                dynamic_assignments[(r, c)] = [torch.sum(x) for x in per_timestep]
        return DynamicMatrix(base_mat, dynamic_assignments)

    # utils ------------------------------------------
    def _get_key(self, kwargs: dict) -> Tuple[str, str]:
        extra = set(kwargs).difference({self.dim1_name, self.dim2_name})
        if extra:
            raise TypeError(f"Unexpected argument(s): {extra}.")
        return kwargs.get(self.dim1_name), kwargs.get(self.dim2_name)

    @classmethod
    def _rename_merged_key(cls, old_key: Tuple[str, str], process_name: str) -> Tuple:
        new_key = list(old_key)
        for i, d in enumerate(new_key, start=1):
            if not cls._is_measure_dim(i):
                new_key[i] = (process_name, old_key[i])
        return tuple(new_key)

    def _check_or_expand_dims(self, *args):
        assert len(args) == 2
        for i, arg in enumerate(args, start=1):
            dim_names = getattr(self, f'dim{i}_names')
            if arg not in dim_names:
                if self._is_measure_dim(i):
                    dim_names.append(arg)
                else:
                    raise ValueError(f"{arg} not in {dim_names}")

    @classmethod
    def _is_measure_dim(cls, i: int) -> bool:
        return getattr(cls, f'dim{i}_name') == 'measure'

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

    def _validate_adjustment(self, value: DesignMatAdjustment, check_slow_grad: bool) -> DesignMatAdjustment:
        if isinstance(value, Tensor):
            self._check_adjust_tens(value, in_list=False, check_slow_grad=check_slow_grad)
        elif _is_dynamic_assignment(value):
            if len(value) != self.num_timesteps:
                raise ValueError(
                    f"The adjustment is a sequence w/len {len(value)}, but batch num_timesteps is {self.num_timesteps}"
                )
            for tens in value:
                self._check_adjust_tens(tens, in_list=True, check_slow_grad=check_slow_grad)
        else:
            raise ValueError("Expected `value` be list/tuple or tensor")

        return value

    def _check_adjust_tens(self, tens: Tensor, in_list: bool, check_slow_grad: bool = True):
        is_scalar = tens.numel() == 1
        is_num_groups_1d = list(tens.shape) == [self.num_groups]
        if not (is_scalar or is_num_groups_1d):
            if in_list:
                raise ValueError("If list is passed, then each element should be scalar or be 1D w/len = num_groups.")
            else:
                raise ValueError("If tensor is passed, then should be scalar or be 1D w/len = num_groups.")
        if in_list and check_slow_grad and is_slow_grad(tens):
            raise RuntimeError(
                f"This adjustment appears to have been generated by first creating a tensor that requires_grad, then "
                f"splitting it into a list of tensors, one for each time-point. This way of creating adjustments should"
                f" be avoided because it leads to a very slow backwards pass; instead make the list of tensors first, "
                f"*then* do computations that require_grad on each element of the list. If this is a false-alarm, avoid"
                f" this error by passing check_slow_grad=False to the adjustment method."
            )


class TransitionMatrix(DesignMatrix):
    # TODO: order?
    dim1_name = 'from_element'
    dim2_name = 'to_element'

    @property
    def from_elements(self):
        return self.dim1_names

    @property
    def to_elements(self):
        return self.dim2_names


class MeasureMatrix(DesignMatrix):
    # TODO: order?
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
    A diagonal-only matrix that will be multiplied by the process-variance.
    """
    dim1_name = 'state_element'
    dim2_name = 'state_element'

    def __init__(self, dim_names: Sequence[str]):
        super().__init__(dim1_names=dim_names, dim2_names=dim_names)

    def assign(self, value: DesignMatAssignment, **kwargs):
        if value != 0.0:
            raise ValueError(f"Cannot override assignment-value for {self.__class__.__name__}.")
        super().assign(value=value, **kwargs)

    def set_ilink(self, ilink: Optional[Callable], **kwargs):
        if ilink is not torch.exp:
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


def _is_dynamic_assignment(x) -> bool:
    return isinstance(x, (list, tuple))
