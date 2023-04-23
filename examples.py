import torch
from torch import nn

from sparse_snn.SparseLinear import SparseDense
from sparse_snn.SparseZO import LocalLIF
from sparse_snn.SparseLinear.try_sparse_grad import set_elements_to_zero


class RandomSampler:
    def __init__(self):
        pass

    def __call__(self, sample_number: int, input_size: torch.Size, device):
        return torch.randn((sample_number,) + input_size).to(device)

# example for construct multi-layer SNN
class MultiLayerSNN(nn.Module):

    def __init__(self):
        super(MultiLayerSNN, self).__init__()
        self.fc1 = SparseDense(16, 32)
        self.snn1 = LocalLIF(0.5, 0.1, 0.1, 5, RandomSampler())
        self.fc2 = SparseDense(32, 16)
        self.snn2 = LocalLIF(0.5, 0.1, 0.1, 5, RandomSampler())
        self.fc3 = SparseDense(16, 10)



    def forward(self, inputs):
        inputs = self.fc1(inputs)
        inputs = self.snn1(inputs)
        inputs = self.fc2(inputs)
        inputs = self.snn2(inputs)
        inputs = self.fc3(inputs)
        return inputs


if __name__ == '__main__':
    net = MultiLayerSNN()
    inputs = torch.randn(100, 64, 16)
    set_elements_to_zero(inputs, 0.5)
    inputs = inputs.to_sparse_coo()
    outputs = net(inputs)
    print(outputs)



