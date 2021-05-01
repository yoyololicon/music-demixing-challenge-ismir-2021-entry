import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.Spectrogram import STFT

from utils import add_weight_norms


class _NonCausalLayer2D(nn.Module):
    def __init__(self,
                 dilation,
                 aux_channels,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 kernel_size,
                 bias,
                 flipped=False,
                 last_layer=False):
        super().__init__()
        self.h_pad_size = dilation[1] * (kernel_size[1] - 1)
        self.pad_size = dilation[0] * (kernel_size[0] - 1) // 2
        self.flipped = flipped

        self.V = nn.Conv1d(aux_channels, dilation_channels * 2, kernel_size[0],
                           dilation=dilation[0], padding=self.pad_size, bias=bias)
        self.W = nn.Conv2d(residual_channels, dilation_channels * 2,
                           kernel_size=kernel_size, padding=(
                               self.pad_size, 0),
                           dilation=dilation, bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv2d(
                dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv2d(
                dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        buf = F.pad(x, [0, self.h_pad_size]
                    if self.flipped else [self.h_pad_size, 0])
        xy = self.W(buf) + self.V(y).unsqueeze(3)
        zw, zf = xy.chunk(2, 1)
        z = zw.tanh().mul(zf.sigmoid())
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output.add(x), skip
        else:
            return None, skip

    def inverse_forward(self, x, y, buffer=None):
        if buffer is None:
            buffer = F.pad(x, [0, self.h_pad_size]
                           if self.flipped else [self.h_pad_size, 0])
        else:
            buffer = torch.cat(
                (x, buffer[..., :-1]), -1) if self.flipped else torch.cat((buffer[..., 1:], x), -1)
        xy = self.W(buffer) + self.V(y).unsqueeze(3)
        zw, zf = xy.chunk(2, 1)
        z = zw.tanh().mul(zf.sigmoid())
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output.add(x), skip, buffer
        else:
            return None, skip, buffer


class WN2D(nn.Module):
    def __init__(self,
                 layers,
                 in_channels,
                 aux_channels,
                 dilation_channels=64,
                 residual_channels=64,
                 skip_channels=64,
                 flipped=False,
                 bias=False):
        super().__init__()

        self.dilations = [(3 ** d, 2 ** d) for d in range(layers)]
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.aux_chs = aux_channels

        self.start = nn.Conv2d(in_channels, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(_NonCausalLayer2D(d,
                                                      aux_channels,
                                                      dilation_channels,
                                                      residual_channels,
                                                      skip_channels,
                                                      (3, 2),
                                                      bias, flipped=flipped) for d in self.dilations)
        self.layers.append(_NonCausalLayer2D(self.dilations[-1],
                                             aux_channels,
                                             dilation_channels,
                                             residual_channels,
                                             skip_channels,
                                             (3, 2),
                                             bias,
                                             flipped=flipped,
                                             last_layer=True))
        self.layers.apply(add_weight_norms)

        self.end = nn.Conv2d(skip_channels, in_channels * 2, 1, bias=bias)
        self.end.weight.data.zero_()
        if bias:
            self.end.bias.data.zero_()

    def forward(self, x, y):
        x = self.start(x)
        cum_skip = None
        for layer in self.layers:
            x, skip = layer(x, y)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)

    def inverse_forward(self, x, y, buffer_list=None):
        x = self.start(x)
        new_buffer_list = []
        if buffer_list is None:
            buffer_list = [None] * len(self.layers)

        cum_skip = None
        for layer, buf in zip(self.layers, buffer_list):
            x, skip, buf = layer.inverse_forward(x, y, buf)
            new_buffer_list.append(buf)
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip = cum_skip + skip

        return self.end(cum_skip).chunk(2, 1) + (new_buffer_list,)


class WaveFlow(nn.Module):
    def __init__(self,
                 flows,
                 layers,
                 n_group,
                 n_fft=4096,
                 hop_length=1024,
                 channels=2,
                 **kwargs):
        super().__init__()
        self.flows = flows
        self.layers = layers
        self.n_group = n_group
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.upsample_rate = hop_length // n_group
        self.channels = channels
        self.register_buffer('window', torch.hann_window(n_fft))

        self.WNs = nn.ModuleList()

        for i in range(self.flows):
            self.WNs.append(WN2D(self.layers, channels * 4,
                                 n_fft // 4 * channels, flipped=i % 2 == 1, **kwargs))

        self.spec = nn.Sequential(
            nn.ReflectionPad1d((n_fft // 2 - hop_length // 2,
                                n_fft // 2 + hop_length // 2)),
            STFT(n_fft, freq_bins=n_fft // 4, hop_length=hop_length, freq_scale='log', fmin=40, fmax=16000,
                 sr=44100, output_format='Magnitude')
        )

    def get_spec(self, x):
        spec = self.spec(x.view(-1, 1, x.shape[2])).add_(1e-7).log_()
        return F.interpolate(spec, scale_factor=self.upsample_rate, mode='linear').view(x.shape[0], x.shape[1] * spec.shape[1], -1)

    def forward(self, x, h):
        seq_len = x.shape[3]
        pad_size = seq_len % self.n_group
        if pad_size:
            x = F.pad(x, [0, self.n_group - pad_size])

        origin_shape = x.shape
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], -1, self.n_group)
        assert x.shape[2] <= h.shape[2]
        h = h[..., :x.shape[2]]

        for k, WN in enumerate(self.WNs):
            if k % 2:
                padded_x = F.pad(x[..., 1:], [0, 1])
            else:
                padded_x = F.pad(x[..., :-1], [1, 0])
            log_s, t = WN(padded_x, h)
            x = x * log_s.exp() + t

            if k:
                logdet += log_s.sum((1, 2, 3))
            else:
                logdet = log_s.sum((1, 2, 3))

        return x.view(*origin_shape[:3], -1)[..., :seq_len], logdet

    def inverse(self, z, h):
        seq_len = z.shape[3]
        pad_size = seq_len % self.n_group
        if pad_size:
            z = F.pad(z, [0, self.n_group - pad_size])

        origin_shape = z.shape
        z = z.view(z.shape[0], z.shape[1] * z.shape[2], -1, self.n_group)
        assert z.shape[2] <= h.shape[2]
        h = h[..., :z.shape[2]]

        logdet = None
        for k, WN in zip(range(self.flows - 1, -1, -1), self.WNs[::-1]):
            xnew = torch.zeros_like(z[..., :1])
            x = []

            buffer_list = None
            for i in range(self.n_group):
                if k % 2:
                    index = slice(self.n_group - i - 1, self.n_group - i)
                else:
                    index = slice(i, i+1)
                log_s, t, buffer_list = WN.inverse_forward(
                    xnew, h, buffer_list)
                xnew = (z[..., index] - t) / log_s.exp()
                x.append(xnew)

                if logdet is None:
                    logdet = log_s.sum((1, 2, 3))
                else:
                    logdet += log_s.sum((1, 2, 3))
            if k % 2:
                x = x[::-1]
            z = torch.cat(x, -1)

        z = z.view(*origin_shape[:3], -1)[..., :seq_len]
        return z, -logdet

    @torch.no_grad()
    def infer(self, h, sigma=1.):

        z = h.new_empty((h.shape[0], 4, self.channels,
                         h.shape[-1] * self.n_group)).normal_(std=sigma)
        x, _ = self.inverse(z, h)
        return x, _


if __name__ == "__main__":
    y = torch.randn(1, 2, 44100)
    x = torch.ones(1, 4, 2, 44100)
    model = WaveFlow(6, 5, 32, n_fft=1024, hop_length=256,
                     dilation_channels=64,
                     residual_channels=64,
                     skip_channels=64)

    spec = model.get_spec(y)
    print(spec.shape)
    with torch.no_grad():
        z, logdet = model(x, spec)
        new_x, logdet = model.inverse(z, spec)
    print(z.shape, logdet, new_x)
    assert torch.allclose(new_x, x), torch.max((new_x - x).abs())
