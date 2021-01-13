from typing import Tuple, Sequence, List, Dict, Optional, Iterable

import torch

from torch import nn, Tensor, jit


class Process(nn.Module):
    # https://github.com/pytorch/pytorch/issues/42885
    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 h_module: Optional[nn.Module] = None,
                 h_tensor: Optional[Tensor] = None,
                 f_modules: Optional[nn.ModuleDict] = None,
                 f_tensors: Optional[Dict[str, Tensor]] = None):
        super(Process, self).__init__()
        self.id = id

        # state elements:
        self.state_elements = state_elements
        self.se_to_idx = {se: i for i, se in enumerate(self.state_elements)}

        # the initial mean of the state elements
        self.init_mean = nn.Parameter(.1 * torch.randn(len(self.state_elements)))

        # observation matrix:
        if (int(h_module is None) + int(h_tensor is None)) != 1:
            raise TypeError("Exactly one of `h_module`, `h_tensor` must be passed.")
        self.h_module = h_module
        self.h_tensor = h_tensor
        self.h_kwarg = ''

        # transition matrix:
        self.f_modules = f_modules
        self.f_tensors = f_tensors
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

    @property
    def expected_kwargs(self) -> List[str]:
        out: List[str] = []
        if self.f_kwarg != '':
            out.append(self.f_kwarg)
        if self.h_kwarg != '' and self.h_kwarg not in out:
            out.append(self.h_kwarg)
        return out

    @jit.unused
    def get_all_expected_kwargs(self) -> Iterable[str]:
        return (x for x in list(self.expected_kwargs) + list(self.expected_init_mean_kwargs or []))

    @jit.unused
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
        # if self.id == 'lm':
        #     self.debugger()
        if self.h_module is None:
            assert self.h_tensor is not None
            return self.h_tensor
        else:
            return self.h_module() if input is None else self.h_module(input)

    def f_forward(self, input: Optional[Tensor]) -> Tensor:
        F = torch.zeros(len(self.state_elements), len(self.state_elements))
        if self.f_tensors is not None:
            for from__to, tens in self.f_tensors.items():
                assert tens is not None
                from_el, sep, to_el = from__to.partition("->")
                c = self.se_to_idx[from_el]
                if sep == '':
                    F[:, c] = tens
                else:
                    r = self.se_to_idx[to_el]
                    F[r:(r + 1), c] = tens
        if self.f_modules is not None:
            for from__to, module in self.f_modules.items():
                from_el, sep, to_el = from__to.partition("->")
                c = self.se_to_idx[from_el]
                tens = module() if input is None else module(input)
                if sep == '':
                    F[:, c] = tens
                else:
                    r = self.se_to_idx[to_el]
                    F[r:(r + 1), c] = tens
        return F
