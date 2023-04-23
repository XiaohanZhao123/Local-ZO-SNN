import torch
from typing import Any


class OutputDenseInputSparse(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x) -> Any:
        return x

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        return grad_outputs.to_dense()  # test if we can receive sparse gradient


class InputDenseOutputSparse(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x) -> Any:
        return x

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        return grad_outputs.to_sparse()  # test if we can pass sparse gradient


def test_case1():
    x = torch.randn((32, 16, 16), requires_grad=True)
    y = OutputDenseInputSparse.apply(x)
    z = InputDenseOutputSparse.apply(y)
    z.sum().backward()
    print(y.grad, z.grad, x.grad)


if __name__ == '__main__':
    test_case1()
