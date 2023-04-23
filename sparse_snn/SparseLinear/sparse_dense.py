from typing import Any, Tuple

import torch


class SparceForwardSparseBackward(torch.autograd.Function):
    """Sparse forward and sparse backward function, will not restore the tensor to dense mode."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> Any:
        """
        compute the forward propagation of the sparse linear layer, the input should be sparse tensor.

        Args:
            ctx: context for gradient computation
            input: input tensor, should be in shape [time, batch, features] and sparse
            weight: weight tensor, should be in shape [input_features, output_features]
            bias: bias tensor, should be in shape [output_features]

        Returns: a strid tensor in shape [time, batch, output_features]

        """
        assert input.is_sparse is True, 'input must be sparse, got dense tensor instead'
        assert input.dim() == 3, 'input must be in shape [time, batch, features]'
        ctx.save_for_backward(input, weight, bias)
        return torch.stack([torch.sparse.mm(input[i], weight) + bias for i in range(input.shape[0])])


    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass