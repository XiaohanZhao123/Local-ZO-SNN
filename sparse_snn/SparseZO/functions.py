import torch
from typing import Any


class LocalLIFLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, inputs, z, u_th: float, beta: float, sigma: float) -> Any:
        """
        forward with local zo approximation for gradient

        Args:
            ctx: the autodiff context
            inputs: the input tensor, should be in shape [time, batch, features]
            z: randomly sampled directions, should be in shape [time, sample_num, *inputs[1:]]
            u_th: the threshold of output spike
            beta: drop rate from previous state
            sigma: for gradient approximation

        Returns:

        """
        state, output = 0, 0
        outputs = []
        grads = []
        for input, z in zip(inputs, z):
            state = - u_th * output + state * beta + input
            output = (state > u_th).float()
            outputs.append(output)
            grad = torch.mean((state < torch.abs(z) * sigma + u_th).float() * z, dim=0) / (2 * sigma)
            grads.append(grad)

        ctx.constant = [u_th, beta, sigma]
        outputs = torch.stack(outputs).to_sparse_coo()
        grads = torch.flip(torch.stack(grads), (0,)).to_sparse_coo()
        ctx.save_for_backward(grads)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # expect dense grad_outputs
        u_th, beta, sigma = ctx.constant
        grads,  = ctx.saved_tensors
        grad_states = []
        grad_state_prev = torch.zeros_like(grad_outputs[0]).to_sparse_coo()  # the gradient of previous input
        grad_outputs_flip = torch.flip(grad_outputs, (0,))
        # a backward indexing
        for grad_output, grad in zip(grad_outputs_flip, grads):
            grad_output_new = grad_output - grad_state_prev * u_th  # sparse + dense = dense
            grad_state_current = torch.mul(grad, grad_output_new) + grad_state_prev * beta  # sparse + sparse = sparse
            grad_states.append(grad_state_current)
            grad_state_prev = grad_state_current

        # return grad states directly since d[gradient_state]/d[inputs] = 1
        return torch.stack(grad_states), None, None, None, None
