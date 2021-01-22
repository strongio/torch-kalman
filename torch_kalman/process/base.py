from typing import Tuple, Sequence, List, Dict, Optional, Iterable

import torch

from torch import nn, Tensor, jit
from torch_kalman.internals.utils import get_owned_kwarg


class Process(nn.Module):
    """
    The Process is defined by the state-elements it generates predictions for. It generates three kinds of predictions:
    (1) the initial state, (2) the observation matrix, (3) the transition matrix.
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
                 init_mean_kwargs: Optional[List[str]] = None,
                 time_varying_kwargs: Optional[List[str]] = None,
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
        :param init_mean_kwargs: TODO
        :param time_varying_kwargs: TODO
        :param no_pcov_state_elements: TODO
        :param no_icov_state_elements: TODO
        """

        super(Process, self).__init__()
        self.id = id

        # state elements:
        self.state_elements = state_elements
        self.se_to_idx = {se: i for i, se in enumerate(self.state_elements)}
        assert len(state_elements) == len(self.se_to_idx), f"state-elements are not unique:{state_elements}"

        # the initial mean of the state elements
        self.init_mean = nn.Parameter(.1 * torch.randn(len(self.state_elements)))

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

        #
        self.expected_init_mean_kwargs: Optional[List[str]] = init_mean_kwargs
        self.time_varying_kwargs: Optional[List[str]] = time_varying_kwargs

    def get_initial_state_mean(self, input: Optional[Dict[str, Tensor]] = None) -> Tensor:
        assert input is None or len(input) == 0  # not used by base class
        assert self.expected_init_mean_kwargs is None  # this method should be overridden if there are
        return self.init_mean

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, str, Tensor]]:
        if self.expected_init_mean_kwargs:
            for key in self.expected_init_mean_kwargs:
                found_key, value = get_owned_kwarg(self.id, key, kwargs)
                yield found_key, key, 'init_mean', torch.as_tensor(value)
        for key in [self.f_kwarg, self.h_kwarg]:
            if key == '':
                continue
            found_key, value = get_owned_kwarg(self.id, key, kwargs)
            key_type = 'time_varying' if key in (self.time_varying_kwargs or []) else 'static'
            yield found_key, key, key_type, torch.as_tensor(value)

    def forward(self,
                inputs: Dict[str, Tensor],
                which: str,
                cache: Dict[str, Tensor]) -> Tensor:
        if '_empty' not in cache:
            # jit doesn't like Optional[Tensor] in some situations
            cache['_empty'] = torch.empty(0)

        if which == 'h':
            h_input = cache['_empty'] if self.h_kwarg == '' else inputs[self.h_kwarg]
            h_key = self._get_cache_key(self.h_kwarg, h_input, prefix=f'{self.id}_h')
            if h_key is not None:
                if h_key not in cache:
                    cache[h_key] = self.h_forward(h_input)
                return cache[h_key]
            else:
                return self.h_forward(h_input)

        # F
        if which == 'f':
            f_input = cache['_empty'] if self.f_kwarg == '' else inputs[self.f_kwarg]
            f_key = self._get_cache_key(self.f_kwarg, f_input, prefix=f'{self.id}_f')
            if f_key is not None:
                if f_key not in cache:
                    cache[f_key] = self.f_forward(f_input)
                return cache[f_key]
            else:
                return self.f_forward(f_input)

        else:
            raise RuntimeError(f"Unrecognized which='{which}'.")

    def _get_cache_key(self, kwarg: str, input: Optional[Tensor], prefix: str) -> Optional[str]:
        """
        Subclasses could use `input` to determine the cache-key
        """
        if self.time_varying_kwargs is not None:
            if kwarg in self.time_varying_kwargs:
                return None
        return f'{prefix}_static'

    def h_forward(self, input: Tensor) -> Tensor:
        if self.h_module is None:
            assert self.h_tensor is not None
            H = self.h_tensor
        else:
            H = self.h_module(input)
        if not self._validate_h_shape(H):
            msg = (
                f"`Process(id='{self.id}').h_forward()` produced output with shape {H.shape}, "
                f"but expected ({len(self.state_elements)},) or (num_groups, {len(self.state_elements)})."
            )
            if input is not None:
                msg += f" Input had shape {input.shape}."
            raise RuntimeError(msg)
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

    def f_forward(self, input: Tensor) -> Tensor:
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
            assert len(tens.shape) == 1 or len(tens.shape) == 2 and tens.shape[-1] == 1
            F[:, r, c] = tens
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
