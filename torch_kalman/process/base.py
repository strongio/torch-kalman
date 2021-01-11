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

        # transition matrix:
        self.f_modules = f_modules
        self.f_tensors = f_tensors

        # fillted in by set_measure:
        self.measure = ''

        # elements without process covariance, defaults to none
        self.no_pcov_state_elements: List[str] = []

        #
        self.cache: Dict[str, Tensor] = {'null': torch.empty(0)}  # jit doesn't like empty

    def get_groupwise_kwargs(self, *args, **kwargs) -> Dict[str, Tensor]:
        raise NotImplementedError

    def get_timewise_kwargs(self, *args, **kwargs) -> Dict[str, Tensor]:
        raise NotImplementedError

    def set_measure(self, measure: str) -> 'Process':
        self.measure = measure
        return self

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        H = self.h_forward(inputs)
        F = self.f_forward(inputs)
        return H, F

    def h_forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        # TODO: caching
        if self.h_module is None:
            assert self.h_tensor is not None
            return self.h_tensor
        else:
            return self.h_module(inputs)

    def f_forward(self, inputs: Dict[str, Tensor]) -> Tensor:
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
            tens = module(inputs)
            if sep == '':
                F[:, c] = tens
            else:
                r = self.se_to_idx[to_el]
                F[r:(r + 1), c] = tens
        return F
