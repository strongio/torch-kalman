from typing import Tuple, Sequence, List, Dict, Optional

import torch

from torch import nn, Tensor


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
        self.measure = ''

        # elements without process covariance, defaults to none
        self.no_pcov_state_elements: List[str] = []

        #
        self.cache: Optional[Dict[str, Tensor]] = None

        #
        self.expected_init_state_kwargs = ['']  # jit doesn't like empty list

    @property
    def expected_kwargs(self) -> List[str]:
        out: List[str] = []
        if self.f_kwarg != '':
            out.append(self.f_kwarg)
        if self.h_kwarg != '':
            out.append(self.h_kwarg)
        return out

    def enable_cache(self, enable: bool = True):
        if enable:
            self.cache = {'_null': torch.empty(0)}  # jit doesn't like empty dict
        else:
            self.cache = None

    def set_measure(self, measure: str) -> 'Process':
        self.measure = measure
        return self

    def forward(self, inputs: Dict[str, Tensor], tv_kwargs: List[str]) -> Tuple[Tensor, Tensor]:
        # H
        h_input = None if self.h_kwarg == '' else inputs[self.h_kwarg]
        if self.h_kwarg not in tv_kwargs and self.cache is not None:
            if 'static_h' not in self.cache:
                self.cache['static_h'] = self.h_forward(h_input)
            H = self.cache['static_h']
        else:
            H = self.h_forward(h_input)

        # F
        f_input = None if self.f_kwarg == '' else inputs[self.f_kwarg]
        if self.f_kwarg not in tv_kwargs and self.cache is not None:
            if 'static_f' not in self.cache:
                self.cache['static_f'] = self.f_forward(f_input)
            F = self.cache['static_f']
        else:
            F = self.f_forward(f_input)
        return H, F

    def h_forward(self, input: Optional[Tensor]) -> Tensor:
        if self.h_module is None:
            assert self.h_tensor is not None
            return self.h_tensor
        else:
            return self.h_module() if input is None else self.h_module(input)

    def f_forward(self, input: Optional[Tensor]) -> Tensor:
        F = torch.zeros(len(self.state_elements), len(self.state_elements))
        for from__to, tens in self.f_tensors.items():
            from_el, sep, to_el = from__to.partition("->")
            c = self.se_to_idx[from_el]
            if sep == '':
                F[:, c] = tens
            else:
                r = self.se_to_idx[to_el]
                F[r:(r + 1), c] = tens
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
