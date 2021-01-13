from typing import Tuple, Sequence, List, Dict, Optional, Iterable

import torch

from torch import nn, Tensor, jit


class Process(nn.Module):
    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 h_module: Optional[nn.Module] = None,
                 h_tensor: Optional[Tensor] = None,
                 f_modules: Optional[nn.ModuleDict] = None,
                 f_tensors: Optional[Dict[str, Tensor]] = None):
        """
        :param id: Unique identifier for the process
        :param state_elements: List of strings with the state-element names
        :param h_module: A torch.nn.Module which, when called (default with no input; can be overridden in subclasses
        with self.h_kwarg), will produce the 'observation' matrix: a XXXX. Only one of h_module or h_tensor should be
        passed.
        :param h_tensor: A tensor that is the 'observation' matrix (see `h_module`). Only one of h_module or h_tensor
        should be  passed.
        :param f_modules: A torch.nn.ModuleDict; each element specifying a transition between state-elements. The keys
        specify the state-elements in the format '{from_el}->{to_el}'. The values are torch.nn.Modules which, when
        called (default with no input; can be overridden in subclasses with self.f_kwarg), will produce that element
        for the transition matrix.
        :param f_tensors: A dictionary of tensors, specifying elements of the F-matrix. See `f_modules` for key format.
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
        self.h_kwarg = ''

        # transition matrix:
        self.f_tensors = f_tensors
        self.f_modules = f_modules
        self.f_kwarg = ''

        # filled in by set_measure:
        self.measure: str = ''

        # elements without process covariance, defaults to none
        self.no_pcov_state_elements: List[str] = []
        # elements without initial covariance, defaults to none:
        self.no_icov_state_elements: List[str] = []

        #
        self.cache: Dict[str, Tensor] = {'_null': torch.empty(0)}  # jit dislikes empty dicts
        self._cache_enabled = False

        #
        self.expected_init_mean_kwargs: Optional[List[str]] = None
        self.time_varying_kwargs: Optional[List[str]] = None

    def get_initial_state_mean(self, input: Optional[Dict[str, Tensor]] = None) -> Tensor:
        assert input is None or len(input) == 0  # not used by base class
        assert self.expected_init_mean_kwargs is None  # this method should be overridden if there are
        return self.init_mean

    @jit.ignore
    def get_all_expected_kwargs(self) -> Iterable[str]:
        out = set()
        if self.expected_init_mean_kwargs:
            for x in self.expected_init_mean_kwargs:
                out.add(x)
        if self.f_kwarg != '':
            out.add(self.f_kwarg)
        if self.h_kwarg != '':
            out.add(self.h_kwarg)
        return out

    @jit.ignore
    def enable_cache(self, enable: bool = True):
        if enable:
            self.cache.clear()
        self._cache_enabled = enable

    def set_measure(self, measure: str) -> 'Process':
        self.measure = measure
        return self

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        # H
        h_input = None if self.h_kwarg == '' else inputs[self.h_kwarg]
        h_key = self._get_cache_key(self.h_kwarg, h_input, prefix='h')
        if self._cache_enabled and h_key is not None:
            if h_key not in self.cache:
                self.cache[h_key] = self.h_forward(h_input)
            H = self.cache[h_key]
        else:
            H = self.h_forward(h_input)

        # F
        f_input = None if self.f_kwarg == '' else inputs[self.f_kwarg]
        f_key = self._get_cache_key(self.f_kwarg, f_input, prefix='f')
        if self._cache_enabled and f_key is not None:
            if f_key not in self.cache:
                self.cache[f_key] = self.f_forward(f_input)
            F = self.cache[f_key]
        else:
            F = self.f_forward(f_input)
        return H, F

    def _get_cache_key(self, kwarg: str, input: Optional[Tensor], prefix: str) -> Optional[str]:
        """
        Subclasses could use `input` to determine the cache-key
        """
        if self.time_varying_kwargs is not None:
            if kwarg in self.time_varying_kwargs:
                return None
        return f'{prefix}_static'

    def h_forward(self, input: Optional[Tensor]) -> Tensor:
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
        orig_h = H

        valid = True
        if len(H.shape) > 3:
            valid = False
        else:
            if len(H.shape) == 3:
                if H.shape[-1] == 1:
                    H = H.squeeze(-1)  # handle in next case
                else:
                    valid = False
            if len(H.shape) == 1:
                if len(self.state_elements) == 1:
                    H = H.unsqueeze(-1)  # handle in next case
                elif H.shape[0] != len(self.state_elements):
                    valid = False
            if len(H.shape) == 2:
                if H.shape[-1] != len(self.state_elements):
                    valid = False
        return valid

    def f_forward(self, input: Optional[Tensor]) -> Tensor:
        #
        assignments: List[Tuple[Tuple[int, int], Tensor]] = []
        num_groups = 1
        if self.f_tensors is not None:
            for from__to, tens in self.f_tensors.items():
                assert tens is not None
                from_el, sep, to_el = from__to.partition("->")
                c = self.se_to_idx[from_el]
                r = self.se_to_idx[to_el]
                if len(tens.shape) > 1:
                    assert len(tens.shape) == 2 or len(tens.shape) == 2 and tens.shape[-1] == 1 and tens.shape[0] > 1
                    num_groups = tens.shape[0]
                assignments.append(((r, c), tens))
        if self.f_modules is not None:
            for from__to, module in self.f_modules.items():
                from_el, sep, to_el = from__to.partition("->")
                c = self.se_to_idx[from_el]
                r = self.se_to_idx[to_el]
                tens = module(input)
                if len(tens.shape) > 1:
                    assert len(tens.shape) == 2 or len(tens.shape) == 2 and tens.shape[-1] == 1 and tens.shape[0] > 1
                    num_groups = tens.shape[0]
                assignments.append(((r, c), tens))
        F = torch.zeros(num_groups, len(self.state_elements), len(self.state_elements))
        for (r, c), tens in assignments:
            F[:, r, c] = tens
        return F
