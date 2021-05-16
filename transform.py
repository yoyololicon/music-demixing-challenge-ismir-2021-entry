import numpy as np
import random


class RandomSwapLR(object):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        assert 0 <= p <= 1, "invalid probability value"
        self.p = p

    def __call__(self, stems: np.ndarray):
        tmp = np.flip(stems, 1)
        for i in range(stems.shape[0]):
            if random.random() < self.p:
                stems[i] = tmp[i]
        return stems


class RandomGain(object):
    def __init__(self, low=0.25, high=1.25) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, stems):
        gains = np.random.uniform(self.low, self.high, stems.shape[0])
        stems = stems * gains[:, None, None]
        return stems
