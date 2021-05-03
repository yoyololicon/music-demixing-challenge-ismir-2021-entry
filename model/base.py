import torch
import torch.nn as nn


class Spec(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x, inverse=False):
        shape = x.shape
        if inverse:
            spec = x.reshape(-1, *shape[-2:])
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.n_fft,
                self.window,
            ).view(*shape[:-2], -1)
        x = x.reshape(-1, x.shape[-1])
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            self.window,
            return_complex=True
        )
        # (batch, channels, nfft // 2 + 1, frames)
        return spec.view(*shape[:-1], *spec.shape[-2:])
