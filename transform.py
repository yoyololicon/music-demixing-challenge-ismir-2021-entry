import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from julius import ResampleFrac
import random
from model import Spec
from torchaudio.transforms import TimeStretch
from torchaudio.functional import phase_vocoder


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


class RandomFlipPhase(RandomSwapLR):
    def __call__(self, stems: np.ndarray):
        """
        Args:
            stems (np.array): (Num_sources, Num_channels, L)
        Return:
            stems (np.array): (Num_sources, Num_channels, L)
        """
        for i in range(stems.shape[0]):
            if random.random() < self.p:
                stems[i] *= -1
        return stems


class _DeviceTransformBase(nn.Module):
    def __init__(self, rand_size, p=0.2):
        super().__init__()
        self.p = p
        self.rand_size = rand_size

    def _transform(self, stems, index):
        raise NotImplementedError

    def forward(self, stems: torch.Tensor):
        """
        Args:
            stems (torch.Tensor): (B, Num_sources, Num_channels, L)
        Return:
            perturbed_stems (torch.Tensor): (B, Num_sources, Num_channels, L')
        """
        shape = stems.shape
        orig_len = shape[-1]
        stems = stems.view(-1, *shape[-2:])
        select_mask = torch.rand(stems.shape[0], device=stems.device) < self.p
        if not torch.any(select_mask):
            return stems.view(*shape)

        select_idx = torch.where(select_mask)[0]
        perturbed_stems = torch.zeros_like(stems)
        perturbed_stems[~select_mask] = stems[~select_mask]
        selected_stems = stems[select_mask]
        rand_idx = torch.randint(
            self.rand_size, (selected_stems.shape[0],), device=stems.device)

        for i in range(self.rand_size):
            mask = rand_idx == i
            if not torch.any(mask):
                continue
            masked_stems = selected_stems[mask]
            perturbed_audio = self._transform(masked_stems, i)

            diff = perturbed_audio.shape[-1] - orig_len

            put_idx = select_idx[mask]
            if diff >= 0:
                perturbed_stems[put_idx] = perturbed_audio[..., :orig_len]
            else:
                perturbed_stems[put_idx, :, :orig_len+diff] = perturbed_audio

        perturbed_stems = perturbed_stems.view(*shape)
        return perturbed_stems


class SpeedPerturb(_DeviceTransformBase):
    def __init__(
        self, orig_freq=44100, speeds=[90, 100, 110], **kwargs
    ):
        super().__init__(len(speeds), **kwargs)
        self.orig_freq = orig_freq
        self.resamplers = nn.ModuleList()
        self.speeds = speeds
        for s in self.speeds:
            new_freq = self.orig_freq * s // 100
            self.resamplers.append(ResampleFrac(self.orig_freq, new_freq))

    def _transform(self, stems, index):
        y = self.resamplers[index](
            stems.view(-1, stems.shape[-1])).view(*stems.shape[:-1], -1)
        return y


class RandomPitch(_DeviceTransformBase):
    def __init__(
        self, semitones=[-2, -1, 0, 1, 2], n_fft=2048, hop_length=512, **kwargs
    ):
        super().__init__(len(semitones), **kwargs)
        self.resamplers = nn.ModuleList()

        semitones = torch.tensor(semitones, dtype=torch.float32)
        rates = 2 ** (-semitones / 12)
        rrates = rates.reciprocal()
        rrates = (rrates * 100).long()
        rrates[rrates % 2 == 1] += 1
        rates = 100 / rrates

        self.register_buffer('rates', rates)
        self.spec = Spec(n_fft, hop_length)
        self.stretcher = TimeStretch(hop_length, n_freq=n_fft // 2 + 1)

        for rr in rrates.tolist():
            self.resamplers.append(ResampleFrac(rr, 100))

    def _transform(self, stems, index):
        spec = torch.view_as_real(self.spec(stems))
        stretched_spec = self.stretcher(spec, self.rates[index])
        stretched_stems = self.spec(
            torch.view_as_complex(stretched_spec), inverse=True)
        shifted_stems = self.resamplers[index](
            stretched_stems.view(-1, stretched_stems.shape[-1])).view(*stretched_stems.shape[:-1], -1)
        return shifted_stems


if __name__ == "__main__":
    trsfn = nn.Sequential(
        SpeedPerturb(), RandomPitch()
    )

    x = torch.randn(4, 4, 2, 22050)
    y = trsfn(x)
    print(y.shape, x[0, 0, 0, -100:], y[0, 0, 0, -100:])
