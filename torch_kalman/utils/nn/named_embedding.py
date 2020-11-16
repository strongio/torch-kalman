from typing import Sequence, Dict
from warnings import warn

import torch
from torch.nn import Embedding
import numpy as np


class NamedEmbedding(Embedding):
    max_name_len = 64

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
        self._names_as_ints = torch.nn.Parameter(torch.zeros((self.num_embeddings, self.max_name_len)))
        self._names_as_ints.requires_grad_(False)
        self._name_to_idx = None

    @property
    def name_to_idx(self) -> Dict[str, int]:
        """
        We'd like to store `name_to_idx` in the state so that it's saved, and loaded with load_state_dict. But this is
        only for Parameters -- and can't use a ParameterDict because load_state_dict will complain about mismatched
        keys. So we convert the names to integers then save them in a 2d Parameter.
        """
        if self._name_to_idx is None:
            self._name_to_idx = {}
            for i, int_repr in enumerate(self._names_as_ints.tolist()):
                if set(int_repr) == {0.0}:
                    # unassigned
                    continue
                self._name_to_idx[_ints_to_str(int_repr).rstrip()] = i
        return self._name_to_idx

    def _input_to_idx(self, input: Sequence[str]) -> torch.Tensor:
        indices = []
        for nm in input:
            nm = str(nm).rstrip()
            if nm not in self.name_to_idx:
                if not self.training:
                    indices.append(-1)
                    continue
                if len(self.name_to_idx) >= self.num_embeddings:
                    raise RuntimeError(
                        f"Got a new group name '{nm}', but all {self.num_embeddings} idx are taken by previous "
                        f"groups. If trying to predict for new groups set `self.train(False)`"
                    )
                else:
                    self._names_as_ints[len(self.name_to_idx)] = torch.tensor(_str_to_ints(nm, width=self.max_name_len))
                    self._name_to_idx = None  # reset `name_to_idx` cache
            indices.append(self.name_to_idx[nm])

        with torch.no_grad():
            idx = torch.tensor(indices, dtype=torch.long)
        return idx

    def forward(self, input: Sequence[str]) -> torch.Tensor:
        idx = self._input_to_idx(input)
        out = torch.zeros((len(input), self.embedding_dim))
        if (idx >= 0).any():
            legal_idx = np.where(idx >= 0)[0]
            out[legal_idx] = super().forward(idx[legal_idx])
        return out

    def reset_parameters(self):
        super().reset_parameters()
        self.weight.data *= .10


def _str_to_ints(astr: str, width: int) -> Sequence[int]:
    if len(astr) > width:
        warn(f"'{astr}' has more characters than {width}, will be truncated")
    return [ord(astr[i]) if i < len(astr) else 32 for i in range(width)]


def _ints_to_str(ints: Sequence) -> str:
    return "".join(str(chr(int(x))) for x in ints)
