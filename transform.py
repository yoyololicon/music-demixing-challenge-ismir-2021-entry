import torch
from torchvision import transforms
import random


class RandomSwapLR(torch.nn.Module):
    def __init__(self, p=0.5, has_all=True) -> None:
        super().__init__()
        assert 0 <= p <= 1, "invalid probability value"
        self.p = p
        self.all = has_all

    def forward(self, sample):
        x, y = sample

        if y.dim == x.dim:
            if random.random() < self.p:
                y = y.flip(0)
                x = x.flip(0)
        elif self.all:
            for i in range(y.shape[0]):
                if random.random() < self.p:
                    y[i] = y[i].flip(0)
            x = y.sum(0)
        else:
            y = y.flip(1)
            x = x.flip(0)
        return x, y


class RandomGain(torch.nn.Module):
    def __init__(self, low=0.25, high=1.25, has_all=True) -> None:
        super().__init__()
        self.low = low
        self.high = high
        self.all = has_all

    def forward(self, sample):
        x, y = sample

        if y.dim == x.dim or not self.all:
            gain = random.uniform(self.low, self.high)
            y *= gain
            x *= gain
        else:
            for i in range(y.shape[0]):
                gain = random.uniform(self.low, self.high)
                y[i] *= gain
            x = y.sum(0)

        return x, y
