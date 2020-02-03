from typing import Optional, Sequence, Hashable

import torch
from torch.nn import Embedding


class NamedEmbedding(Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, dev_code: bool = False, *args, **kwargs):
        if dev_code:
            raise NotImplementedError  # TODO: support deviation coding
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=None,
            *args,
            **kwargs
        )
        # this is a param dict so it'll get saved in state_dict()
        self._name_to_idx = torch.nn.ParameterDict()

    def _input_to_idx(self, input: Hashable) -> torch.Tensor:
        indices = []
        for nm in input:
            if nm not in self._name_to_idx:
                if len(self._name_to_idx) >= self.num_embeddings:
                    # TODO: support deviation coding
                    raise RuntimeError(
                        f"Got a new group name '{nm}', but all {self.num_embeddings} idx are taken by previous groups."
                    )
                else:
                    # hack to store ints in the param dict:
                    self._name_to_idx[nm] = torch.nn.Parameter(len(self._name_to_idx) * torch.ones(1))
                    self._name_to_idx[nm].requires_grad_(False)
            indices.append(self._name_to_idx[nm])

        with torch.no_grad():
            idx = torch.tensor(indices, dtype=torch.long)
        return idx

    def load_state_dict(self, state_dict, strict=True):
        out = super().load_state_dict(state_dict=state_dict, strict=strict)
        for pname, param in self._name_to_idx.items():
            param.requires_grad_(False)
        return out

    def forward(self, input: Sequence[Hashable]) -> torch.Tensor:
        return super().forward(input=self._input_to_idx(input))

    def reset_parameters(self):
        super().reset_parameters()
        self.weight.data *= .10
