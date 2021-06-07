import torch
from torch import nn
import torch.nn.functional as F
import julius
from torch_fftconv import FFTConv1d


def rescale_conv(reference):
    @torch.no_grad()
    def closure(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            std = m.weight.std()
            scale = (std / reference) ** 0.5
            m.weight.div_(scale)
            if m.bias is not None:
                m.bias.div_(scale)
    return closure


class OLAConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, *args, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, *args, **kwargs)
        self.register_buffer('window', torch.hamming_window(
            kernel_size, periodic=True))

        name = 'weight'
        weight = getattr(self, name)
        del self._parameters[name]
        self.register_parameter(name + '_raw', nn.Parameter(weight.data))

        def _set_weight(module, inputs):
            weight = getattr(module, name + '_raw') * getattr(module, 'window')
            setattr(module, name, weight)

        _set_weight(self, None)

        self.register_forward_pre_hook(_set_weight)


class Demucs(nn.Module):
    def __init__(self,
                 channels=64,
                 depth=6,
                 rescale=0.1,
                 resample=True,
                 kernel_size=8,
                 stride=4,
                 lstm_layers=2):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.channels = channels

        if resample:
            self.up_sample = julius.ResampleFrac(1, 2)
            self.down_sample = julius.ResampleFrac(2, 1)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_channels = 2
        for index in range(depth):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, channels, kernel_size, stride
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(channels, channels * 2, 1),
                    nn.GLU(dim=1)
                )
            )

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 8

            decode = [
                nn.Conv1d(channels, channels * 2, 3, padding=1),
                nn.GLU(dim=1),
                nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)
            ]
            if index > 0:
                decode.append(nn.ReLU(inplace=True))
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels *= 2

        channels = in_channels

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels,
            num_layers=lstm_layers,
            dropout=0,
            bidirectional=True)
        self.lstm_linear = nn.Linear(channels * 2, channels)

        self.apply(rescale_conv(reference=rescale))

    def forward(self, x):
        length = x.size(2)

        if hasattr(self, 'up_sample'):
            x = self.up_sample(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.lstm_linear(x).permute(1, 2, 0)

        for decode in self.decoder:
            skip = saved.pop()
            diff = skip.shape[2] - x.shape[2]

            if diff > 0:
                l_pad = diff // 2
                r_pad = diff - l_pad
                x = F.pad(x, [l_pad, r_pad])
            x = x + skip
            x = decode(x)

        if hasattr(self, 'down_sample'):
            x = self.down_sample(x)

        diff = length - x.shape[2]

        if diff > 0:
            l_pad = diff // 2
            r_pad = diff - l_pad
            x = F.pad(x, [l_pad, r_pad])

        x = x.view(x.size(0), 4, 2, x.size(-1))
        return x


class MultiResBlock(nn.Module):
    __constants__ = ['base']

    base: int = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride, scales=3):
        super().__init__()

        self.layers = nn.ModuleList()
        self.channel_list = []

        remain = out_channels
        i = 0
        for _ in range(scales):
            out_channels = remain // 2
            remain -= out_channels
            self.layers.append(nn.Conv1d(in_channels, out_channels,
                                         kernel_size * self.base ** i, stride * self.base ** i))
            i += 1
            self.channel_list.append(out_channels)
        self.layers.append(
            nn.Conv1d(in_channels, remain,
                      kernel_size * self.base ** i, stride * self.base ** i)
        )
        self.channel_list.append(remain)

    def forward(self, x):
        out = []
        for layer in self.layers:
            out.append(layer(x))

        length = out[0].shape[2]
        base = 1
        for i in range(1, len(out)):
            base *= self.base
            y = out[i]
            y = y.repeat_interleave(base, dim=2)
            diff = length - y.shape[2]
            if diff > 0:
                l_pad = diff // 2
                y = F.pad(y, [l_pad, diff - l_pad])
            out[i] = y
        return torch.cat(out, 1)


class MultiResTransposeBlock(nn.Module):
    base: int = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride, scales=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.channel_list = []

        remain = out_channels
        i = 0
        for _ in range(scales):
            out_channels = remain // 2
            remain -= out_channels
            self.layers.append(nn.ConvTranspose1d(in_channels, out_channels,
                                                  kernel_size * self.base ** i, stride * self.base ** i))
            i += 1
            self.channel_list.append(out_channels)
        self.layers.append(
            nn.ConvTranspose1d(in_channels, remain,
                               kernel_size * self.base ** i, stride * self.base ** i)
        )
        self.channel_list.append(remain)

    def forward(self, x):
        out = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = F.avg_pool1d(x, self.base, self.base)
            out.append(layer(x))

        length = out[0].shape[2]
        for i in range(1, len(out)):
            y = out[i]
            diff = y.shape[2] - length
            if diff > 0:
                l_truc = diff // 2
                r_truc = diff - l_truc
                y = y[..., l_truc:-r_truc]
            out[i] = y
        return torch.cat(out, 1)


class D2ResBlock(nn.Module):
    base: int = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride, scales=3):
        super().__init__()

        self.layers = nn.ModuleList()
        self.channel_list = []

        remain = out_channels
        i = 0
        for _ in range(scales):
            self.layers.append(nn.Conv1d(in_channels, remain,
                                         kernel_size, dilation=self.base ** i,
                                         padding=(kernel_size - 1) * self.base ** i // 2 if i > 0 else 0))
            in_channels = remain // 2
            remain -= in_channels
            i += 1
            self.channel_list.append(in_channels)
        self.layers.append(
            nn.Conv1d(in_channels, remain,
                      kernel_size, dilation=self.base ** i,
                      padding=(kernel_size - 1) * self.base ** i // 2 if i > 0 else 0)
        )
        self.channel_list.append(remain)
        self.final_avg = nn.AvgPool1d(stride, stride)

    def forward(self, x):
        out = []
        skips = []
        for i, layer in enumerate(self.layers):
            tmp = layer(x).split(self.channel_list[i:], 1)
            x = tmp[0]

            tmp = tmp[1:]
            if i > 0:
                x = x + skips.pop(0)
                skips = [s + t for s, t in zip(skips, tmp)]
            else:
                skips = list(tmp)
            x = x.relu()
            out.append(x)

        length = out[0].shape[2]
        for i in range(1, len(out)):
            y = out[i]
            diff = y.shape[2] - length
            if diff > 0:
                l_truc = diff // 2
                r_truc = diff - l_truc
                y = y[..., l_truc:-r_truc]
            out[i] = y
        return self.final_avg(torch.cat(out, 1))


class Demucs2(nn.Module):
    def __init__(self,
                 channels=64,
                 depth=6,
                 rescale=0.1,
                 resample=True,
                 kernel_size=8,
                 stride=4,
                 lstm_layers=3,
                 scales=3):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.channels = channels

        if resample:
            self.up_sample = julius.ResampleFrac(1, 2)
            self.down_sample = julius.ResampleFrac(2, 1)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_channels = 2
        for index in range(depth):
            self.encoder.append(
                nn.Sequential(
                    MultiResBlock(in_channels, channels,
                                  kernel_size, stride, scales=scales),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(channels, channels * 2, 1),
                    nn.GLU(dim=1)
                )
            )

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 8

            decode = [
                nn.Conv1d(channels, channels * 2, 3, padding=1),
                nn.GLU(dim=1),
                nn.ConvTranspose1d(channels, out_channels,
                                   kernel_size, stride)
            ]
            if index > 0:
                decode.append(nn.ReLU(inplace=True))
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels *= 2

        channels = in_channels

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels // 2,
            num_layers=lstm_layers,
            dropout=0,
            bidirectional=True)

        self.apply(rescale_conv(reference=rescale))

    def forward(self, x):
        length = x.size(2)

        if hasattr(self, 'up_sample'):
            x = self.up_sample(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = x.permute(1, 2, 0)

        for decode in self.decoder:
            skip = saved.pop()
            diff = skip.shape[2] - x.shape[2]

            if diff > 0:
                l_pad = diff // 2
                r_pad = diff - l_pad
                x = F.pad(x, [l_pad, r_pad])
            x = x + skip
            x = decode(x)

        if hasattr(self, 'down_sample'):
            x = self.down_sample(x)

        diff = length - x.shape[2]

        if diff > 0:
            l_pad = diff // 2
            r_pad = diff - l_pad
            x = F.pad(x, [l_pad, r_pad])

        x = x.view(x.size(0), 4, 2, x.size(-1))
        return x


if __name__ == "__main__":
    net = Demucs2(channels=32)  # .cuda()
    # net = D2ResBlock(2, 32, 8, 4, scales=3)
    # net = torch.jit.script(net)
    print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.rand(1, 2, 44100 * 2)  # .cuda()
    y = net(x)
    print(y.shape)
