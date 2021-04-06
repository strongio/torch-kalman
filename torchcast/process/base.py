from typing import Tuple, Sequence, List, Dict, Optional, Iterable

import torch
from numpy import ndarray

from torch import nn, Tensor, jit
from torchcast.internals.utils import get_owned_kwarg


class Process(nn.Module):
    """
    The Process is defined by the state-elements it generates predictions for. It generates two kinds of predictions:
    (1) the observation matrix, (2) the transition matrix.
    """

    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 measure: Optional[str] = None,
                 h_module: Optional[nn.Module] = None,
                 h_tensor: Optional[Tensor] = None,
                 h_kwarg: str = '',
                 f_modules: Optional[nn.ModuleDict] = None,
                 f_tensors: Optional[Dict[str, Tensor]] = None,
                 f_kwarg: str = '',
                 no_pcov_state_elements: Optional[List[str]] = None,
                 no_icov_state_elements: Optional[List[str]] = None):
        """

        :param id: Unique identifier for the process
        :param state_elements: List of strings with the state-element names
        :param measure: The name of the measure for this process.
        :param h_module: A torch.nn.Module which, when called (default with no input; can be overridden in subclasses
        with self.h_kwarg), will produce the 'observation' matrix: a XXXX. Only one of h_module or h_tensor should be
        passed.
        :param h_tensor: A tensor that is the 'observation' matrix (see `h_module`). Only one of h_module or h_tensor
        should be  passed.
        :param h_kwarg: TODO
        :param f_modules: A torch.nn.ModuleDict; each element specifying a transition between state-elements. The keys
        specify the state-elements in the format '{from_el}->{to_el}'. The values are torch.nn.Modules which, when
        called (default with no input; can be overridden in subclasses with self.f_kwarg), will produce that element
        for the transition matrix. Additionally, the key can be 'all_self', in which case the output should have
        `shape[-1] == len(state_elements)`; this allows specifying the transition of each state-element to itself with
        a single call.
        :param f_tensors: A dictionary of tensors, specifying elements of the F-matrix. See `f_modules` for key format.
        :param f_kwarg: TODO
        :param no_pcov_state_elements: TODO
        :param no_icov_state_elements: TODO
        """

        super(Process, self).__init__()
        self.id = id

        # state elements:
        self.state_elements = state_elements
        self.se_to_idx = {se: i for i, se in enumerate(self.state_elements)}
        assert len(state_elements) == len(self.se_to_idx), f"state-elements are not unique:{state_elements}"

        # observation matrix:
        if (int(h_module is None) + int(h_tensor is None)) != 1:
            raise TypeError("Exactly one of `h_module`, `h_tensor` must be passed.")
        self.h_module = h_module
        self.h_tensor = h_tensor
        self.h_kwarg = h_kwarg

        # transition matrix:
        self.f_tensors = f_tensors
        if isinstance(f_modules, dict):
            f_modules = nn.ModuleDict(f_modules)
        self.f_modules = f_modules
        self.f_kwarg = f_kwarg

        # can be populated later, as long as its before torch.jit.script
        self.measure: str = '' if measure is None else measure

        # elements without process covariance, defaults to none
        self.no_pcov_state_elements: Optional[List[str]] = no_pcov_state_elements
        # elements without initial covariance, defaults to none:
        self.no_icov_state_elements: Optional[List[str]] = no_icov_state_elements

    @jit.ignore
    def offset_initial_state(self, initial_state: Tensor, start_offsets: Optional[Sequence] = None) -> Tensor:
        return initial_state

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, Optional[Tensor]]]:
        for key in [self.f_kwarg, self.h_kwarg]:
            if key == '':
                continue
            found_key, value = get_owned_kwarg(self.id, key, kwargs)
            if value is not None:
                value = torch.as_tensor(value)
            yield found_key, key, value

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return self.h_forward(inputs.get(self.h_kwarg)), self.f_forward(inputs.get(self.f_kwarg))

    def h_forward(self, input: Optional[Tensor]) -> Tensor:
        if self.h_module is None:
            assert self.h_tensor is not None
            H = self.h_tensor
        elif self.h_kwarg != '':
            assert input is not None
            H = self.h_module(input)
        else:
            if torch.jit.is_scripting():
                raise NotImplementedError("h_modules that do not take inputs are currently unsupported for JIT")
            H = self.h_module()
        if not self._validate_h_shape(H):
            msg = (
                f"`Process(id='{self.id}').h_forward()` produced output with shape {H.shape}, "
                f"but expected ({len(self.state_elements)},) or (num_groups, {len(self.state_elements)})."
            )
            if input is not None:
                msg += f" Input had shape {input.shape}."
            raise RuntimeError(msg)
        if torch.isnan(H).any() or torch.isinf(H).any():
            raise RuntimeError(f"{self.id} produced H with nans")
        return H

    def _validate_h_shape(self, H: torch.Tensor) -> bool:
        # H should be:
        # - (num_groups, state_size, 1)
        # - (num_groups, state_size)
        # - (state_size, 1)
        # - (state_size, )
        if len(H.shape) > 3:
            return False
        else:
            if len(H.shape) == 3:
                if H.shape[-1] == 1:
                    H = H.squeeze(-1)  # handle in next case
                else:
                    return False
            if len(H.shape) == 1:
                if len(self.state_elements) == 1:
                    H = H.unsqueeze(-1)  # handle in next case
                elif H.shape[0] != len(self.state_elements):
                    return False
            if len(H.shape) == 2:
                if H.shape[-1] != len(self.state_elements):
                    return False
        return True

    def f_forward(self, input: Optional[Tensor]) -> Tensor:
        diag: Optional[Tensor] = None
        assignments: List[Tuple[Tuple[int, int], Tensor]] = []

        # in first pass, convert keys to (r,c)s in the F-matrix, and establish the batch dim:
        num_groups = 1
        if self.f_tensors is not None:
            for from__to, tens in self.f_tensors.items():
                assert tens is not None
                rc = self._transition_key_to_rc(from__to)
                if len(tens.shape) > 1:
                    assert num_groups == 1 or num_groups == tens.shape[0]
                    num_groups = tens.shape[0]
                if rc is None:
                    assert diag is None
                    diag = tens
                else:
                    assignments.append((rc, tens))
        if self.f_modules is not None:
            for from__to, module in self.f_modules.items():
                rc = self._transition_key_to_rc(from__to)
                tens = module(input)
                # TODO: this should technically do `if f_kwarg=='': tens=module()` but this breaks JIT
                if len(tens.shape) > 1:
                    assert num_groups == 1 or num_groups == tens.shape[0]
                    num_groups = tens.shape[0]
                if rc is None:
                    assert diag is None
                    diag = tens
                else:
                    assignments.append((rc, tens))

        # in the second pass, create the F-matrix and assign (r,c)s:
        state_size = len(self.state_elements)
        F = torch.zeros(num_groups, state_size, state_size)
        # common application is diagonal F, efficient to store/assign that as one
        if diag is not None:
            if diag.shape[-1] != state_size:
                assert len(diag.shape) == 1 and diag.shape[0] == 1
                diag_mat = diag * torch.eye(state_size)
            else:
                diag_mat = torch.diag_embed(diag)
                assert F.shape[-2:] == diag_mat.shape[-2:]
            F = F + diag_mat
        # otherwise, go element-by-element:
        for (r, c), tens in assignments:
            if diag is not None:
                assert r != c, "cannot have transitions from {se}->{same-se} if `all_self` transition was used."
            if len(tens.shape) == 2:
                assert tens.shape[-1] == 1
                tens = tens.squeeze(-1)
            else:
                assert len(tens.shape) <= 1
            F[:, r, c] = tens

        if torch.isnan(F).any() or torch.isinf(F).any():
            raise RuntimeError(f"{self.id} produced F with nans")
        return F

    def _transition_key_to_rc(self, transition_key: str) -> Optional[Tuple[int, int]]:
        from_el, sep, to_el = transition_key.partition("->")
        if sep == '':
            assert from_el == 'all_self', f"Expected '[from_el]->[to_el]', or 'all_self'. Got '{transition_key}'"
            return None
        else:
            c = self.se_to_idx[from_el]
            r = self.se_to_idx[to_el]
            return r, c

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id={repr(self.id)})'
