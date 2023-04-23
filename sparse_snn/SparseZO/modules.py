import torch
from .functions import LocalLIFLayer
from typing import Callable, Tuple


class LocalLIF(torch.nn.Module):

    def __init__(self, u_th: float, beta: float, sigma: float, sample_num: int,
                 random_sampler: Callable[[int, torch.Size, torch.device], torch.Tensor], sample_number: int = 5):
        """
        local FIL layer for spike sequence encoding, backward with sparse zo

        Args:
            u_th: the threshold voltage
            beta: the descent rate
            sigma: the sigma for zo approximation
            sample_num: the number of random samples for approximate gradient
            random_sampler:  the specific random sampler for generate random directions
            sample_number: the number of samples for each input
        """
        super(LocalLIF, self).__init__()
        self.u_th = u_th
        self.beta = beta
        self.sigma = sigma
        self.sample_num = sample_num
        self.random_sampler = random_sampler

    def forward(self, inputs: torch.Tensor):
        z = self.random_sampler(self.sample_num, inputs.shape, inputs.device)
        return LocalLIFLayer.apply(inputs, z, self.u_th, self.beta, self.sigma)
