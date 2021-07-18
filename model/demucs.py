import torch
from torch import nn
import torch.nn.functional as F
import julius

from .cfp import MLC, CFP


@torch.jit.script
def glu(a, b):
    return a * b.sigmoid()


@torch.jit.script
def standardize(x, mu, std):
    return (x - mu) / std


@torch.jit.script
def destandardize(x, mu, std):
    return x * std + mu


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
            hidden_size=channels // 2,
            num_layers=lstm_layers,
            dropout=0,
            bidirectional=True)
        self.lstm_linear = nn.Linear(channels, channels)

        self.apply(rescale_conv(reference=rescale))

    def forward(self, x):
        length = x.size(2)

        mono = x.mean(1, keepdim=True)
        mu = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True).add_(1e-5)
        x = (x - mu) / std

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
            # diff = skip.shape[2] - x.shape[2]

            # if diff > 0:
            #     l_pad = diff // 2
            #     r_pad = diff - l_pad
            #     x = F.pad(x, [l_pad, r_pad])
            x = x + skip[..., :x.shape[2]]
            x = decode(x)

        if hasattr(self, 'down_sample'):
            x = self.down_sample(x)

        # diff = length - x.shape[2]

        # if diff > 0:
        #     l_pad = diff // 2
        #     r_pad = diff - l_pad
        #     x = F.pad(x, [l_pad, r_pad])

        x = x * std + mu
        x = x.view(x.size(0), 4, 2, x.size(-1))
        return x


class DemucsSplit(nn.Module):
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
        self.convs_1x1 = nn.ModuleList()
        self.dec_pre_convs = nn.ModuleList()
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
                out_channels = in_channels * 2
            else:
                out_channels = 8

            self.convs_1x1.insert(0,
                                  nn.Conv1d(channels, channels * 4, 3, padding=1, bias=False))
            self.dec_pre_convs.insert(0,
                                      nn.Conv1d(channels * 2, channels * 4, 3, padding=1, groups=4))
            decode = [
                nn.ConvTranspose1d(channels * 2, out_channels,
                                   kernel_size, stride, groups=4)
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
        self.lstm_linear = nn.Linear(channels * 2, channels * 2)

        self.apply(rescale_conv(reference=rescale))

    def forward(self, x):
        length = x.size(2)

        mono = x.mean(1, keepdim=True)
        mu = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True).add_(1e-5)
        x = (x - mu) / std

        if hasattr(self, 'up_sample'):
            x = self.up_sample(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.lstm_linear(x).permute(1, 2, 0)

        for decode, pre_dec, conv1x1 in zip(self.decoder, self.dec_pre_convs, self.convs_1x1):
            skip = saved.pop()
            # diff = skip.shape[2] - x.shape[2]

            # if diff > 0:
            #     l_pad = diff // 2
            #     r_pad = diff - l_pad
            #     x = F.pad(x, [l_pad, r_pad])

            x = pre_dec(x) + conv1x1(skip[..., :x.shape[2]])
            a, b = x.view(x.shape[0], 4, -1, x.shape[2]).chunk(2, 2)
            x = a * b.sigmoid()
            x = decode(x.view(x.shape[0], -1, x.shape[3]))

        if hasattr(self, 'down_sample'):
            x = self.down_sample(x)

        # diff = length - x.shape[2]

        # if diff > 0:
        #     l_pad = diff // 2
        #     r_pad = diff - l_pad
        #     x = F.pad(x, [l_pad, r_pad])

        x = x * std + mu
        x = x.view(-1, 4, 2, x.size(-1))
        return x


class DemucsCFP(nn.Module):
    def __init__(self,
                 channels=48,
                 depth=6,
                 rescale=0.1,
                 resample=True,
                 kernel_size=8,
                 stride=4,
                 lstm_layers=2,
                 hop_size_depth=4,
                 n_fft=8192,
                 g=[0.1, 0.9, 0.9, 0.7, 0.8, 0.5],
                 hipass_f=80,
                 lowpass_t=1000 / 800,
                 start_midi=40,
                 end_midi=84,
                 division=4,
                 norm=True
                 ):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.hop_size_depth = hop_size_depth

        if resample:
            self.up_sample = julius.ResampleFrac(1, 2)
            self.down_sample = julius.ResampleFrac(2, 1)

        self.encoder = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.dec_pre_convs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        condition_encoder = []

        num_pitch = (end_midi - start_midi + 1) * division

        in_channels = 2
        condition_channels = num_pitch
        for index in range(depth):
            if index >= hop_size_depth:
                condition_encoder += [
                    nn.Conv1d(
                        condition_channels,
                        condition_channels * 2,
                        kernel_size,
                        stride
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        condition_channels * 2,
                        condition_channels * 4,
                        1
                    ),
                    nn.GLU(dim=1)
                ]
                condition_channels *= 2

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
                out_channels = in_channels * 2
            else:
                out_channels = 8

            self.convs_1x1.insert(0,
                                  nn.Conv1d(channels, channels * 4, 3, padding=1, bias=False))
            self.dec_pre_convs.insert(0,
                                      nn.Conv1d(channels * 2, channels * 4, 3, padding=1, groups=4))
            decode = [
                nn.ConvTranspose1d(channels * 2, out_channels,
                                   kernel_size, stride, groups=4)
            ]
            if index > 0:
                decode.append(nn.ReLU(inplace=True))
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels *= 2

        self.condition_encoder = nn.Sequential(
            *condition_encoder
        )
        channels = in_channels

        self.lstm = nn.LSTM(
            input_size=channels + condition_channels * 2,
            hidden_size=channels,
            num_layers=lstm_layers,
            dropout=0,
            bidirectional=True)
        self.lstm_linear = nn.Linear(
            channels * 2 + condition_channels * 2, channels * 2)

        self.apply(rescale_conv(reference=rescale))

        hop_length = 2 * stride ** (hop_size_depth - 1)
        context = sum([(kernel_size - 1) * stride **
                       i for i in range(hop_size_depth)]) + 1

        self.pad_size = [(n_fft - context) // 2,  (n_fft) // 2]

        self.mlc = MLC(44100, n_fft, hop_length, g=g,
                       hipass_f=hipass_f, lowpass_t=lowpass_t)
        self.cfp = CFP(n_fft, 44100, start_midi=start_midi,
                       end_midi=end_midi, division=division, norm=norm)

    def forward(self, x):
        length = x.size(2)
        batch = x.size(0)

        mono = x.mean(1, keepdim=True)
        mu = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True).add_(1e-5)
        # x = (x - mu) / std
        x = standardize(x, mu, std)

        ceps, spec = self.mlc(F.pad(x.view(-1, length), self.pad_size))
        cfp = self.cfp(ceps, spec)
        condition = self.condition_encoder(cfp)
        condition = condition.view(batch, -1, condition.shape[2])

        if hasattr(self, 'up_sample'):
            x = self.up_sample(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        x = torch.cat([x, condition], 1)
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = torch.cat([x, condition.permute(2, 0, 1)], 2)
        x = self.lstm_linear(x).permute(1, 2, 0)

        for decode, pre_dec, conv1x1 in zip(self.decoder, self.dec_pre_convs, self.convs_1x1):
            skip = saved.pop()
            x = pre_dec(x) + conv1x1(skip[..., :x.shape[2]])
            a, b = x.view(x.shape[0], 4, -1, x.shape[2]).chunk(2, 2)
            # x = a * b.sigmoid()
            x = glu(a, b)
            x = decode(x.view(x.shape[0], -1, x.shape[3]))

        if hasattr(self, 'down_sample'):
            x = self.down_sample(x)

        # x = x * std + mu
        x = destandardize(x, mu, std)
        x = x.view(-1, 4, 2, x.size(-1))
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
    from torchinfo import summary
    net = DemucsCFP(channels=32)  # .cuda()
    # net = D2ResBlock(2, 32, 8, 4, scales=3)
    # net = torch.jit.script(net)
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.rand(1, 2, 44100 * 2)  # .cuda()
    summary(net, input_data=x, device='cpu',
            col_names=("input_size", "output_size", "num_params", "kernel_size",
                       "mult_adds"))
