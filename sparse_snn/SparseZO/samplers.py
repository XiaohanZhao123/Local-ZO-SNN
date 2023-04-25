import torch
from abc import ABC, abstractmethod


class Sampler(ABC):
    """basic class for sample random directions, different sampler eventually equals
     to different surrogate gradient functions"""
    @abstractmethod
    def __call__(self, sample_number: int, input_size: torch.Size, device) -> torch.Tensor:
        pass


class RandomSampler(Sampler):
    def __init__(self):
        pass

    def __call__(self, sample_number: int, input_size: torch.Size, device):
        return torch.randn((input_size[0],) + (sample_number,) + input_size[1:]).to(device)
