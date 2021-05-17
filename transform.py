import numpy as np
import torch
import torchaudio
import random


class RandomSwapLR(object):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        assert 0 <= p <= 1, "invalid probability value"
        self.p = p

    def __call__(self, stems: np.ndarray):
        """
        Args:
            stems (np.array): (Num_sources, Num_channels, L)
        Return:
            stems (np.array): (Num_sources, Num_channels, L)
        """
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

    def __call__(self, stems):
        """
        Args:
            stems (np.array): (Num_sources, Num_channels, L)
        Return:
            stems (np.array): (Num_sources, Num_channels, L)
        """
        gains = np.random.uniform(self.low, self.high, stems.shape[0])
        stems = stems * gains[:, None, None]
        return stems


class SpeedPerturb(torch.nn.Module):
    def __init__(
        self, orig_freq=44100, speeds=[90, 100, 110]
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.resamplers = []
        self.speeds = speeds
        for s in self.speeds:
            new_freq = self.orig_freq * s // 100
            self.resamplers.append(
                torchaudio.transforms.Resample(self.orig_freq, new_freq))

    def forward(self, stems):
        """
        Args:
            stems (torch.Tensor): (B, Num_sources, Num_channels, L)
        Return:
            perturbed_stems (torch.Tensor): (B, Num_sources, Num_channels, L')
        """
        # Perform source-wise random perturbation
        new_stems = []
        # init min len
        min_len = 2**32
        for i in range(stems.shape[1]):
            samp_index = torch.randint(len(self.speeds), (1,))[0]
            perturbed_audio = self.resamplers[samp_index](
                stems[:, i].contiguous())
            new_stems.append(perturbed_audio)
            min_len = min(min_len, perturbed_audio.shape[-1])

        perturbed_stems = torch.stack([x[..., :min_len] for x in new_stems], 1)

        return perturbed_stems
