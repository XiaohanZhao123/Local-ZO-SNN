import torch
from spconv import pytorch as spconv
from try_sparse_grad import set_elements_to_zero
from torch.autograd import Function
from typing import Any
from spconv.pytorch import SparseSequential


def test_creating_spase_tensor():
    """
    test how we can successfully create a sparse tensor

    Returns:

    """
    # the features should have shape [N, C], meaning that we have N non-zero elements in C channels
    features = torch.tensor([
        [0, 1, 3],
        [2, 2, 1],
        [3, 4, 5]
    ])  # this matrix means that we have 3 non-zero elements and the tensor has 3 channels

    # the indices should have shape [N, D], which is the indices of this non-zero elements,
    indices = torch.tensor([
        [0, 0, 0],
        [1, 8, 20],
        [20, 6, 80]
    ]).int()  # this matrix means that we have 3 non-zero elements, the batch size is at least 20

    x = spconv.SparseConvTensor(features=features, indices=indices, spatial_shape=[20, 100], batch_size=32)
    print(x)
    print(x.dense())  # converting this one to dense matrix seems ... slow


def test_spase_dense_transform():
    x = torch.randn((32, 16, 16, 3))
    set_elements_to_zero(x, 0.99)
    x = spconv.SparseConvTensor.from_dense(x)
    print(x)


class GetIntermidiateGradient(Function):

    @staticmethod
    def forward(ctx: Any, x:spconv.SparseConvTensor) -> Any:
        # seems that we cannot simply do a in-place update
        output = x.shadow_copy()
        output = output.replace_feature(x.features * 3)
        return output

    @staticmethod
    def backward(ctx: Any, grad_outputs: spconv.SparseConvTensor) -> Any:
        print('whatt?')
        return grad_outputs.replace_feature(grad_outputs.features*3)


def test_get_intermidiate_gradient():
    x = torch.randn((32, 16, 16, 3), )
    set_elements_to_zero(x, 0.99)
    x.requires_grad = True
    x = x.cuda()
    x = spconv.SparseConvTensor.from_dense(x)
    x = GetIntermidiateGradient.apply(x)
    x = spconv.SparseConv2d(3, 3, 3, 1, algo=spconv.ConvAlgo.Native).cuda()(x)
    x = GetIntermidiateGradient.apply(x)
    x = spconv.ToDense()(x)
    x.sum().backward()


if __name__ == '__main__':
    test_get_intermidiate_gradient()
