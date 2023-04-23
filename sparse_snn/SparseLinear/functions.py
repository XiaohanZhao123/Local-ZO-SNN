import torch
from typing import Any


class SparseForwardSparseBackward(torch.autograd.Function):
    """
    compute sparse forward and sparse backward of linear layer for a sequence of input
    """

    @staticmethod
    def forward(ctx: Any, inputs, weight, bias) -> Any:
        assert inputs.is_sparse is True, 'only accept sparse input, got dense'
        ctx.save_for_backward(inputs, weight, bias)
        # don't need to expand bias since we got dense matrix
        return torch.stack([torch.mm(input, weight) + bias for input in inputs])

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # expect sparse grad_outputs
        inputs, weight, bias = ctx.saved_tensors
        inputs = inputs.to_dense()  # transform the format into dense since torch.sparse only support sparse * dense
        grad_w = torch.sum(
            torch.stack([torch.mm(grad_output.t(), input).t() for grad_output, input in zip(grad_outputs, inputs)],
                        dim=0)
        )  # a dense matrix
        grad_b = torch.sum(grad_outputs, dim=(0, 1))  # sparse matrix
        grad_inputs = torch.stack([torch.mm(grad_output, weight.t()) for grad_output in grad_outputs])
        return grad_inputs, grad_w, grad_b.to_dense()  # convert the necessary value into dense


class SparseForwardDenseBackward(torch.autograd.Function):
    """sparse forward and restore the gradient output into dense mode for acceleration of backward"""

    @staticmethod
    def forward(ctx: Any, inputs, weight, bias) -> Any:
        assert inputs.is_sparse is True, 'only accept sparse input, got dense'
        ctx.save_for_backward(inputs, weight, bias)
        # don't need to expand bias since we got dense matrix
        return torch.stack([torch.mm(input, weight) + bias for input in inputs])

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        inputs, weight, bias = ctx.saved_tensors
        inputs: torch.Tensor = inputs.to_dense()  # transform the format into dense since torch.sparse only support sparse * dense
        grad_outputs: torch.Tensor = grad_outputs.to_dense()
        grad_w = torch.sum(torch.bmm(inputs.transpose(0, 2, 1), grad_outputs), dim=0)
        grad_b = torch.sum(grad_outputs, dim=(0,1))
        grad_inputs = torch.mm(grad_outputs, weight.t())
        return grad_inputs, grad_w, grad_b
