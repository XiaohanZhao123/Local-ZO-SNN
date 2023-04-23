from typing import Any, Tuple
from torch import nn
import torch
from .functions import SparseForwardDenseBackward, SparseForwardSparseBackward


class SparseDense(nn.Module):
    def __init__(self, in_features: int, out_features: int, sparse_forward: bool = True, sparse_backward: bool = True,
                 *args, **kwargs):
        """
        the sparse linear class, compute the forward but store the input in sparse format to save memory

        Args:
            in_features:
            out_features:
            sparse_forward:
            sparse_backward:
        """
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        assert sparse_forward is True, 'currently we only support sparse forward'
        if sparse_backward is False:
            self.forward_fn = SparseForwardDenseBackward
        else:
            self.forward_fn = SparseForwardSparseBackward

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_fn.apply(inputs, self.weight, self.bias)
