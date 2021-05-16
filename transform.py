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

    def forward(self, audios):
        """
        Args:
            audios (torch.Tensor): (B, Num_sources, Num_channels, L)
        Return:
            perturbed_audios (torch.Tensor): (B, Num_sources, Num_channels, L')
        """
        # Perform source-wise random perturbation
        new_audios = []
        for i in range(audios.shape[1]):
            samp_index = torch.randint(len(self.speeds), (1,))[0]
            perturbed_audio = self.resamplers[samp_index](audios[:, i].contiguous())
            new_audios.append(perturbed_audio)
            if i == 0:
                min_len = perturbed_audio.shape[-1]
            else:
                if perturbed_audio.shape[-1] < min_len:
                    min_len = perturbed_audio.shape[-1]

        perturbed_audios = torch.zeros(
            audios.shape[0],
            audios.shape[1],
            2,
            min_len,
            device=audios.device,
            dtype=torch.float,
            )

        for i, _ in enumerate(new_audios):
            perturbed_audios[:, i] = new_audios[i][:, :, 0:min_len]

        return perturbed_audios
