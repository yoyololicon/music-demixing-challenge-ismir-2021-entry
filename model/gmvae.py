import torch
from torch import nn
import torch.nn.functional as F
import math
from itertools import chain
from .vae import VAE

import torch.nn.utils.parametrize as parametrize


class GMVAE(VAE):
    def __init__(self,
                 *args,
                 latent_size=512,
                 c=4,
                 **kwargs):
        super().__init__(*args, latent_size=latent_size, **kwargs)

        self.cluster_mu = nn.Parameter(torch.randn(c, latent_size))
        self.cluster_logvar = nn.Parameter(
            torch.full((c, latent_size), math.log(1e-1)), requires_grad=True)

    def forward(self, spec: torch.Tensor):
        return super().forward(spec) + (self.cluster_mu, self.cluster_logvar)


class GMVAEDenoise(GMVAE):
    def __init__(self, *args, latent_size, c, **kwargs):
        super().__init__(*args, latent_size=latent_size, c=c, **kwargs)

        delattr(self, 'affine1')
        delattr(self, 'global_enc')
        delattr(self, 'latent_linear')

        self.new_affine1 = nn.Sequential(
            nn.Conv1d(2 * self.max_bins,
                      self.hidden_channels * 4, 1, bias=False),
            nn.BatchNorm1d(self.hidden_channels * 4),
            nn.Tanh()
        )

        self.bass_enc = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=self.nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.drums_enc = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=self.nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.vocals_enc = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=self.nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.other_enc = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=self.nb_layers,
            dropout=0.4,
            bidirectional=True)

        self.new_latent_linear = nn.Linear(
            self.hidden_channels, self.latent_size * 2)

        self.fix_decoder()

    def fix_decoder(self):
        for p in chain(self.global_dec.parameters(), self.affine2.parameters()):
            p.requires_grad = False
        self.cluster_mu.requires_grad = False
        self.cluster_logvar.requires_grad = False

    def encode(self, spec: torch.Tensor):
        batch, channels, bins, frames = spec.shape
        spec = spec[..., :self.max_bins, :]

        x = spec.reshape(batch, -1, frames)
        x = self.new_affine1(x).view(
            batch, 4, -1, frames).permute(1, 3, 0, 2)
        drums, bass, other, vocals = x

        drums, *_ = self.drums_enc(drums)
        bass, *_ = self.bass_enc(bass)
        other, *_ = self.other_enc(other)
        vocals, *_ = self.vocals_enc(vocals)

        x = torch.stack([drums, bass, other, vocals], 2)
        mu, logvar = self.new_latent_linear(x).chunk(2, 3)

        if self.training:
            z = self.sampling(mu, logvar)
        else:
            z = mu

        return z.transpose(0, 1), mu.transpose(0, 1), logvar.transpose(0, 1)


class LogProb(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, X: torch.Tensor):
        return X.log_softmax(dim=self.dim)

    def right_inverse(self, X):
        return X


class GMVAEJoint(GMVAE):
    def __init__(self, *args, latent_size, c, fix_kc=False, **kwargs):
        super().__init__(*args, latent_size=latent_size, c=c, **kwargs)
        distribution = torch.hann_window(c, periodic=True)
        k_c = [distribution]
        for step in torch.linspace(0, c, 5).tolist()[1:4]:
            k_c.append(
                distribution.roll(int(round(step)), 0)
            )
        k_c.append(torch.ones(c))
        k_c = torch.stack(k_c) + 1e-3
        k_c[:] = 1
        self.log_k_c = nn.Parameter(k_c.log_(), requires_grad=not fix_kc)
        parametrize.register_parametrization(self, 'log_k_c', LogProb())
        # self.log_k_c = torch.full((4, c), -math.log(4 * c))

    def forward(self, spec: torch.Tensor):
        return super().forward(spec) + (self.log_k_c,)


if __name__ == "__main__":
    from torchinfo import summary
    net = GMVAE(max_bins=2000)
    spec = torch.rand(1, 4, 2, 2049, 10)
    summary(net, input_data=spec, col_names=[
            "output_size", "num_params", "input_size"])
