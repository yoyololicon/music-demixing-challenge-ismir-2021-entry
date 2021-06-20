import torch
from torch import nn
import torch.nn.functional as F
import math

import torch.nn.utils.parametrize as parametrize


class LogProb(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, X: torch.Tensor):
        return X.log_softmax(dim=self.dim)

    def right_inverse(self, X):
        return X


class GMVAE(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        hidden_channels=512,
        max_bins=None,
        nb_layers=2,
        latent_size=512,
        c=6
    ):
        super().__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bins:
            self.max_bins = max_bins
        else:
            self.max_bins = self.nb_output_bins
        self.hidden_channels = hidden_channels
        self.n_fft = n_fft
        self.nb_layers = nb_layers
        self.latent_size = latent_size

        self.affine1 = nn.Sequential(
            nn.Conv1d(2 * self.max_bins * 4,
                      hidden_channels * 4, 1, bias=False, groups=4),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.Tanh()
        )

        self.bass_enc = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.drums_enc = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.vocals_enc = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.other_enc = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)

        self.bass_dec = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0,
            bidirectional=True)
        self.drums_dec = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0,
            bidirectional=True)
        self.vocals_dec = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0,
            bidirectional=True)
        self.other_dec = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0,
            bidirectional=True)

        self.affine2 = nn.Sequential(
            nn.Conv1d(hidden_channels * 4,
                      hidden_channels * 4, 1, bias=False, groups=4),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels * 4,
                      2 * self.nb_output_bins * 4, 1, bias=False, groups=4),
            nn.BatchNorm1d(2 * self.nb_output_bins * 4),
        )

        self.latent_linear = nn.Linear(
            self.hidden_channels, self.latent_size + 1)

        self.clusters = nn.Embedding(c, self.latent_size + 1)
        # self.clusters.weight.data[:, -1] = math.log(1e-2)

        self.log_k_c = nn.Parameter(torch.randn(4, c))
        parametrize.register_parametrization(self, 'log_k_c', LogProb())
        # self.log_k_c = torch.full((4, c), -math.log(4 * c))

    def forward(self, spec: torch.Tensor):
        batch, _, channels, bins, frames = spec.shape
        spec = spec[..., :self.max_bins, :]

        x = spec.reshape(batch, -1, frames)
        x = self.affine1(x).view(
            batch, 4, -1, frames).permute(1, 3, 0, 2)
        drums, bass, other, vocals = x

        drums, *_ = self.drums_enc(drums)
        bass, *_ = self.bass_enc(bass)
        other, *_ = self.other_enc(other)
        vocals, *_ = self.vocals_enc(vocals)

        bass_z_mu, bass_z_logvar = self.latent_linear(
            bass).split([self.latent_size, 1], 2)
        drums_z_mu, drums_z_logvar = self.latent_linear(
            drums).split([self.latent_size, 1], 2)
        vocals_z_mu, vocals_z_logvar = self.latent_linear(
            vocals).split([self.latent_size, 1], 2)
        other_z_mu, other_z_logvar = self.latent_linear(
            other).split([self.latent_size, 1], 2)

        bass_z = bass_z_mu + \
            torch.randn_like(bass_z_mu) * torch.exp(0.5 * bass_z_logvar)
        drums_z = drums_z_mu + \
            torch.randn_like(drums_z_mu) * torch.exp(0.5 * drums_z_logvar)
        vocals_z = vocals_z_mu + \
            torch.randn_like(vocals_z_mu) * torch.exp(0.5 * vocals_z_logvar)
        other_z = other_z_mu + \
            torch.randn_like(other_z_mu) * torch.exp(0.5 * other_z_logvar)

        bass, *_ = self.bass_dec(bass_z)
        drums, *_ = self.drums_dec(drums_z)
        other, *_ = self.other_dec(other_z)
        vocals, *_ = self.vocals_dec(vocals_z)

        x = torch.stack([drums, bass, other, vocals], 2).permute(
            1, 2, 3, 0).reshape(batch, -1, frames)
        x = self.affine2(x).view(batch, 4, channels, bins, frames)  # .relu()

        z = torch.stack([drums_z, bass_z, other_z, vocals_z]
                        ).permute(2, 0, 1, 3)
        z_mu = torch.stack([drums_z_mu, bass_z_mu, other_z_mu,
                            vocals_z_mu]).permute(2, 0, 1, 3)
        z_logvar = torch.stack([drums_z_logvar, bass_z_logvar,
                                other_z_logvar, vocals_z_logvar]).permute(2, 0, 1, 3)
        z_c_mu, z_c_logvar = self.clusters.weight.split(
            [self.latent_size, 1], 1)
        log_k_c = self.log_k_c
        return x, z, z_mu, z_logvar, z_c_mu, z_c_logvar, log_k_c


if __name__ == "__main__":
    from torchinfo import summary
    net = GMVAE(max_bins=2000)
    spec = torch.rand(1, 4, 2, 2049, 10)
    summary(net, input_data=spec)
