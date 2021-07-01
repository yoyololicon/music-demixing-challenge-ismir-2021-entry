import torch
from torch import nn


class VAE(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        hidden_channels=512,
        max_bins=None,
        nb_layers=3,
        latent_size=512,
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
            nn.Conv1d(2 * self.max_bins,
                      hidden_channels * 2, 1, bias=False),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Tanh()
        )

        self.global_enc = nn.LSTM(
            input_size=self.hidden_channels * 2,
            hidden_size=self.hidden_channels,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)

        self.global_dec = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_channels,
            num_layers=nb_layers,
            dropout=0,
            bidirectional=True)

        self.affine2 = nn.Sequential(
            nn.Conv1d(hidden_channels * 2,
                      hidden_channels * 2, 1, bias=False),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels * 2,
                      2 * self.nb_output_bins, 1, bias=False),
            nn.BatchNorm1d(2 * self.nb_output_bins),
        )

        self.latent_linear = nn.Linear(
            self.hidden_channels * 2, self.latent_size * 2)

    def sampling(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, spec: torch.Tensor):
        batch, _, channels, bins, frames = spec.shape
        spec = spec[..., :self.max_bins, :]

        x = spec.reshape(batch * 4, -1, frames)
        x = self.affine1(x).permute(2, 0, 1)

        x, *_ = self.global_enc(x)

        mu, logvar = self.latent_linear(
            x.view(frames, batch, 4, -1)).chunk(2, 3)

        if self.training:
            z = self.sampling(mu, logvar)
        else:
            z = mu

        return z.transpose(0, 1), mu.transpose(0, 1), logvar.transpose(0, 1)

    def decode(self, z):
        batch, frames, _, latent_size = z.shape
        x, *_ = self.global_dec(z.transpose(0,
                                            1).reshape(frames, -1, latent_size))

        x = x.permute(1, 2, 0)
        x = self.affine2(x).view(
            batch, 4, 2, self.nb_output_bins, frames).abs()  # .sigmoid()
        return x

    def forward(self, spec: torch.Tensor):
        z, mu, logvar = self.encode(spec)
        xhat = self.decode(z)
        return xhat, z, mu, logvar


if __name__ == "__main__":
    from torchinfo import summary
    net = VAE(max_bins=2000)
    spec = torch.rand(1, 4, 2, 2049, 10)
    summary(net, input_data=spec, col_names=[
            "output_size", "num_params", "input_size"])
