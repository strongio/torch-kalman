import itertools
from typing import Optional, Sequence, Union, Callable, Tuple, List, Dict

import torch
from torch import Tensor
from torch.distributions.utils import broadcast_all

from torch_kalman.internals.batch import Batchable
from torch_kalman.internals.exceptions import InputValidationError
from torch_kalman.internals.utils import bifurcate, is_slow_grad, identity
from torch_kalman.internals.repr import NiceRepr
from torch_kalman.process.utils.design_matrix.utils import DesignMatAssignment, DesignMatAdjustment


class DesignMatrix(NiceRepr, Batchable):
    dim1_name: str = None
    dim2_name: str = None
    _repr_attrs = ('dim1_names', 'dim2_names')

    def __init__(self,
                 dim1_names: Optional[Sequence[str]] = None,
                 dim2_names: Optional[Sequence[str]] = None):
        if dim1_names is None and not self._is_measure_dim(1):
            raise ValueError("Must supply dim1_names")
        self.dim1_names = list(dim1_names or [])

        if dim2_names is None and not self._is_measure_dim(2):
            raise ValueError("Must supply dim2_names")
        self.dim2_names = list(dim2_names or [])

        self._assignments = {}
        self._ilinks = {}

    @property
    def empty(self) -> bool:
        return not self._assignments

    def copy(self) -> 'DesignMatrix':
        return self._from_attributes(
            dim1_names=list(self.dim1_names),
            dim2_names=list(self.dim2_names),
            batch_info=self._batch_info,
            new_assignments=dict(self._assignments),
            new_ilinks=dict(self._ilinks)
        )

    def for_batch(self, num_groups: int, num_timesteps: int) -> 'DesignMatrix':
        if self._batch_info:
            raise RuntimeError("Cannot call `for_batch()` on output of `for_batch()`.")
        out = self.copy()
        out.batch_info = (num_groups, num_timesteps)
        out._ilinks = dict(self._ilinks)
        out._assignments = {}
        for key, values in self._assignments.items():
            # assignments can be no-grad tensors or callables
            out._assignments[key] = [v() if callable(v) else v for v in values]
        return out

    def assign(self, value: DesignMatAssignment, overwrite: bool = False, **kwargs):
        """
        Assign a value to an element of a design-matrix.

        :param value: A float, a (single-element) Tensor, or a callable that produces one of these.
        :param overwrite: If False (default) then cannot re-assign if already assigned; if True will overwrite.
        :param kwargs: The names of the dimensions.
        """
        key = self._get_key(kwargs)
        self._check_or_expand_dims(*key)
        value = self._validate_assignment(value)
        if overwrite or key not in self._assignments:
            self._assignments[key] = [value]
        else:
            raise ValueError(f"Already have assignment for {key}.")

    def set_ilink(self, ilink: Optional[Callable], overwrite: bool = False, **kwargs):
        """
        Set the inverse-link function that will translate value-assignments/adjustments for an element of the
        design-matrix into their final value.

        :param ilink: A callable that is appropriate for torch.Tensors (e.g. torch.exp). If None, then the identity
          link is assumed.
        :param overwrite: If False (default) then cannot re-assign if already assigned; if True will overwrite.
        :param kwargs: The names of the dimensions.
        """
        key = self._get_key(kwargs)
        if key not in self._assignments:
            raise ValueError(f"Tried to set ilink for {key} but must `assign` first.")
        if key in self._ilinks and not overwrite:
            raise ValueError(f"Already have ilink for {key}.")
        assert ilink is None or callable(ilink)
        self._ilinks[key] = ilink

    def adjust(self, value: DesignMatAdjustment, check_slow_grad: bool, **kwargs):
        """
        Adjust the value of an assignment. The final value for an element will be given by (a) taking the sum of the
        initial value and all adjustments, (b) applying the ilink function from `set_ilink()`.

        :param value: Either (a) a torch.Tensor or (b) a sequence of torch.Tensors (one for each timepoint). The tensor
          should be either scalar, or be 1D with length = self.num_groups.
        :param check_slow_grad: A natural way to create adjustments is to first create a tensor that `requires_grad`,
          then split it into a list of tensors, one for each time-point. This way of creating adjustments should be
          avoided because it leads to a very slow backwards pass. When check_slow_grad is True then a heuristic is used
          to check for this "gotcha". It can lead to false-alarms, so disabling is allowed with `check_slow_grad=False`.
        :param kwargs: The names of the dimensions.
        """
        key = self._get_key(kwargs)
        value = self._validate_adjustment(value, check_slow_grad=check_slow_grad)
        try:
            self._assignments[key].append(value)
        except KeyError:
            raise RuntimeError("Tried to adjust {} (in {}); but need to `assign()` it first.".format(key, self))

    @classmethod
    def merge(cls, mats: Sequence[Tuple[str, 'DesignMatrix']]) -> 'DesignMatrix':
        if not all(isinstance(mat, cls) for _, mat in mats):
            raise RuntimeError(f"All mats should be instance of {cls.__name__}")
        dims = set((mat.num_groups, mat.num_timesteps) for _, mat in mats)
        if len(dims) != 1:
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
            new_assignments.update({cls._rename_merged_key(k, process_name): v for k, v in mat._assignments.items()})
            new_ilinks.update({cls._rename_merged_key(k, process_name): v for k, v in mat._ilinks.items()})

        out = cls._from_attributes(
            dim1_names=list(new_dimnames[1]), dim2_names=list(new_dimnames[2]),
            batch_info=list(dims)[0],
            new_assignments=new_assignments,
            new_ilinks=new_ilinks
        )

        return out

    def compile(self) -> 'DynamicMatrix':
        """
        Consolidate assignments then apply the link function. Some assignments can be "frozen" into a pre-computed
        matrix, while others must remain as lists to be evaluated in DesignForBatch as needed.
        """
        from torch_kalman.process.utils.design_matrix.dynamic_matrix import DynamicMatrix

        base_mat = torch.zeros(self.num_groups, len(self.dim1_names), len(self.dim2_names))
        dynamic_assignments = {}
        for (dim1, dim2), values in self._assignments.items():
            r = self.dim1_names.index(dim1)
            c = self.dim2_names.index(dim2)
            ilink = self._ilinks[(dim1, dim2)] or identity
            dynamic, base = bifurcate(values, _is_dynamic_assignment)
            if dynamic:
                # if any dynamic, then all dynamic:
                dynamic = dynamic + [[x] * self.num_timesteps for x in base]
                per_timestep = list(zip(*dynamic))  # invert
                assert len(per_timestep) == self.num_timesteps
                assert (r, c) not in dynamic_assignments.keys()
                dynamic_assignments[(r, c)] = [
                    ilink(torch.sum(torch.stack(broadcast_all(*x), dim=0), dim=0)) for x in per_timestep
                ]
            else:
                base_mat[:, r, c] = ilink(torch.sum(torch.stack(broadcast_all(*base), dim=0), dim=0))

        return DynamicMatrix(base_mat, dynamic_assignments)

    # utils ------------------------------------------
    @classmethod
    def _from_attributes(cls,
                         dim1_names: Sequence,
                         dim2_names: Sequence,
                         batch_info: Tuple,
                         new_assignments: Dict,
                         new_ilinks: Dict
                         ) -> 'DesignMatrix':
        out = cls(dim1_names=dim1_names, dim2_names=dim2_names)
        out._batch_info = batch_info
        out._assignments = new_assignments
        out._ilinks = new_ilinks
        return out

    def _get_key(self, kwargs: dict) -> Tuple[str, str]:
        extra = set(kwargs).difference({self.dim1_name, self.dim2_name})
        if extra:
            raise TypeError(f"Unexpected argument(s): {extra}.")
        return kwargs.get(self.dim1_name), kwargs.get(self.dim2_name)

    @classmethod
    def _rename_merged_key(cls, old_key: Tuple[str, str], process_name: str) -> Tuple:
        new_key = list(old_key)
        for i, d in enumerate(new_key):
            if not cls._is_measure_dim(i + 1):
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
                raise InputValidationError(
                    f"The adjustment is a sequence w/len {len(value)}, but batch num_timesteps is {self.num_timesteps}"
                )
            for tens in value:
                self._check_adjust_tens(tens, in_list=True, check_slow_grad=check_slow_grad)
        else:
            raise InputValidationError("Expected `value` be list/tuple or tensor")

        return value

    def _check_adjust_tens(self, tens: Tensor, in_list: bool, check_slow_grad: bool = True):
        is_scalar = tens.numel() == 1
        is_num_groups_1d = list(tens.shape) == [self.num_groups]
        if not (is_scalar or is_num_groups_1d):
            if in_list:
                raise InputValidationError(
                    "If list is passed, then each element should be scalar or be 1D w/len = num_groups."
                )
            else:
                raise InputValidationError("If tensor is passed, then should be scalar or be 1D w/len = num_groups.")
        if in_list and check_slow_grad and is_slow_grad(tens):
            raise RuntimeError(
                "This adjustment appears to have been generated by first creating a tensor that `requires_grad`, then "
                "splitting it into a list of tensors, one for each time-point. This way of creating adjustments should "
                "be avoided because it leads to a very slow backwards pass; instead make the list of tensors first, "
                "*then* do computations that require_grad on each element of the list. If this is a false-alarm, avoid "
                "this error by passing check_slow_grad=False to the adjustment method."
            )


def _is_dynamic_assignment(x) -> bool:
    return isinstance(x, (list, tuple))
