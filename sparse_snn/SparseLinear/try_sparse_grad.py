import time

import torch
from typing import Any
from matplotlib import pyplot as plt
import numpy as np


class InputSparseOutputDense(torch.autograd.Function):

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


class SNNLyaer(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x) -> Any:
        return x.to_sparse()

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        return grad_outputs.to_sparse()


class SpareLinearLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x) -> Any:
        assert x.is_sparse is True
        return x.to_dense()

    @staticmethod
    def backward(ctx: Any, grad_outputs) -> Any:
        return grad_outputs.to_dense()  # backward a sense tensor


def test_case1():
    """
    test if the sparse gradient can flow through dense tensor

    Returns:

    """
    x = torch.randn((32, 16, 16), requires_grad=True)
    y = InputSparseOutputDense.apply(x)
    z = InputDenseOutputSparse.apply(y)
    z.sum().backward()
    print(y.grad, z.grad, x.grad)


def test_case2():
    """
    test if the dense gradient can flow through sparse tensor

    Returns:

    """
    x = torch.randn((32, 16, 16))
    set_elements_to_zero(x, 0.9)
    x = x.to_sparse_coo()
    x.requires_grad = True
    y = SpareLinearLayer.apply(x)
    z = SNNLyaer.apply(y)
    torch.sparse.sum(z).backward()
    print(y.grad, z.grad, x.grad, x.grad.is_sparse)


def test_case6():
    x = torch.randn((32, 16, 16))
    x.requires_grad = True
    z = SNNLyaer.apply(x)
    z.sum().backward()
    print(x.grad)


def set_elements_to_zero(x: torch.Tensor, proportion):
    """
    helper function to set proportion value to zr

    Args:
        x:
        proportion:0

    Returns: None

    """
    number = torch.numel(x)
    number_to_zero = int(number * proportion)
    indices = torch.randperm(number)[:number_to_zero]
    x.view(-1)[indices] = 0


def test_case4():
    """
    test if we can stack a coo tensor

    Returns:

    """
    x = torch.randn((32, 16, 16))
    set_elements_to_zero(x, 0.9)
    x = x.to_sparse_coo()
    y = torch.stack([x, x, x, x])  # good, we can stack the sparse coo tensor!
    print(y)


def test_case5():
    """
    test some operation between sparse and dense tensors

    Returns:

    """
    x = torch.randn((32, 16, 16))
    set_elements_to_zero(x, 0.9)
    x = x.to_sparse_coo()
    y = torch.randn((32, 16, 16))
    z = x.clone()
    print(f'the result of sparse + sparse is sparse: {(x + z).is_sparse}')
    try:
        print(f'the result of sparse + dense is dense: {not (x + y).is_sparse}')
    except RuntimeError:
        print('pytorch does not support sparse + dense operation')
    print(f'the result of sparse * dense is dense: {not (y + x).is_sparse}')

    b = torch.randn((16))
    print('see if the auto broadcast can happen', b.unsqueeze(0).expand_as(x) + x)
    print('see if we can sum the sparse tensor along two dims', torch.sum(x, dim=(0, 1)))


def test_case7():
    """
    test if we can concat sparse coo tensors

    Returns:

    """
    x = torch.randn((32, 16, 16))
    set_elements_to_zero(x, 0.9)
    x = x.to_sparse_coo()
    y = torch.concat([x] * 20)
    print(y.size(), y)


def test_time():
    x_shape = (2048, 2048)
    w_shape = (2048, 2048)
    x = torch.randn(x_shape)
    w = torch.randn(w_shape)
    set_elements_to_zero(x, 0.9)
    x_sparse = x.to_sparse_coo()
    print(x_sparse.is_coalesced())

    sparse_mul_time = []
    dense_mul_time = []
    sparse_to_dense_time = []

    for _ in range(100):
        start = time.time()
        y = torch.mul(x_sparse, w)
        torch.cuda.synchronize()
        end = time.time()
        sparse_mul_time.append(end - start)

    for _ in range(100):
        start = time.time()
        y = torch.mul(x, w)
        torch.cuda.synchronize()
        end = time.time()
        dense_mul_time.append(end - start)

    for _ in range(100):
        start = time.time()
        z = x_sparse.to_dense()
        y = torch.mul(z, w)
        torch.cuda.synchronize()
        end = time.time()
        sparse_to_dense_time.append(end - start)

    sparse_mul_time = np.mean(sparse_mul_time)
    dense_mul_time = np.mean(dense_mul_time)
    sparse_to_dense_time = np.mean(sparse_to_dense_time)

    plt.figure()
    plt.title('Comparison for time consumption under 100 runs')
    plt.bar([0, 1, 2], [sparse_mul_time, dense_mul_time, sparse_to_dense_time])
    plt.xticks([0, 1, 3], ['Sparse Multiplication Time', 'Dense Mul Time', 'Sparse To Dense Multiplication time'])
    plt.show()


if __name__ == '__main__':
    test_time()
