import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class _Base(nn.Module):
    def get_output_channels(self):
        raise NotImplementedError


class D2_block(_Base):
    __constants__ = [
        'in_channels',
        'k',
        'L',
        'last_N',
    ]

    in_channels: int
    k: int
    L: int
    last_N: int

    def __init__(self,
                 in_channels,
                 k,
                 L,
                 last_n_layers=3):
        super().__init__()

        self.in_channels = in_channels
        self.k = k
        self.L = L
        self.last_N = last_n_layers

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(L):
            self.conv_layers.append(
                nn.Conv2d(
                    k if i > 0 else in_channels,
                    k * (L - i),
                    3,
                    padding=2 ** i,
                    dilation=2 ** i,
                    bias=False
                )
            )

            self.bn_layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(k),
                    nn.ReLU(inplace=True)
                )
            )

    def get_output_channels(self):
        return self.k * min(self.L, self.last_N)

    def forward(self, input: torch.Tensor):
        # the input should be already BN + ReLU before
        outputs = []
        skips = []
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            tmp = conv(input).chunk(self.L - i, 1)
            input = tmp[0]
            tmp = tmp[1:]
            if i > 0:
                input = input + skips.pop(0)
                skips = [s + t for s, t in zip(skips, tmp)]
            else:
                skips = list(tmp)
            input = bn(input)
            outputs.append(input)

        assert len(skips) == 0
        if self.last_N > 1 and len(outputs) > 1:
            return torch.cat(outputs[-self.last_N:], 1)
        return outputs[-1]


class D3_block(_Base):
    def __init__(self,
                 in_channels,
                 M,
                 *args,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.d2_layers = nn.ModuleList()
        concat_channels = in_channels
        for i in range(M):
            self.d2_layers.append(
                D2_block(in_channels, *args, **kwargs)
            )
            concat_channels = self.d2_layers[-1].get_output_channels()
            in_channels += concat_channels

    def get_output_channels(self):
        return self.in_channels + sum(l.get_output_channels() for l in self.d2_layers)

    def forward(self, input):
        raw_inputs = [input]
        for d2 in self.d2_layers:
            input = d2(torch.cat(raw_inputs, 1) if len(
                raw_inputs) > 1 else input)
            raw_inputs.append(input)
        return torch.cat(raw_inputs, 1)


class UNet(_Base):
    def __init__(self,
                 in_channels,
                 down_specs: List[Dict],
                 up_specs: List[Dict]):
        super().__init__()

        assert len(down_specs) == len(up_specs) + 1
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        self.tas_layers = nn.ModuleList()

        skip_channels = []
        for spec in down_specs:
            if len(skip_channels):
                self.transition_layers.append(
                    nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
                    )
                )
                in_channels //= 2
            self.down_layers.append(
                D3_block(in_channels, **spec)
            )
            in_channels = self.down_layers[-1].get_output_channels()
            skip_channels.append(in_channels)
        skip_channels.pop()

        self.register_buffer('tas_kernel', torch.ones(1, 1, 2, 2) * 0.25)

        for spec, skip in zip(up_specs, skip_channels[::-1]):
            self.tas_layers.append(
                nn.ConvTranspose2d(in_channels, in_channels // 2,
                                   1, stride=1, bias=False)
            )
            in_channels //= 2
            self.up_layers.append(
                D3_block(in_channels + skip, **spec)
            )
            in_channels = self.up_layers[-1].get_output_channels()

    def get_output_channels(self):
        return self.up_layers[-1].get_output_channels()

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.down_layers):
            if i:
                x = self.transition_layers[i-1](x)
                if x.shape[2] % 2 or x.shape[3] % 2:
                    x = F.pad(x, [0, x.shape[3] %
                                  2, 0, x.shape[2] % 2], mode='replicate')
                x = F.avg_pool2d(x, 2, 2)
            x = layer(x)
            skips.append(x)
        skips.pop()

        for layer, ts, sk in zip(self.up_layers, self.tas_layers, skips[::-1]):
            x = ts(x)
            # x = x.repeat_interleave(2, 2).repeat_interleave(2, 3)
            x = F.conv_transpose2d(x, self.tas_kernel.expand(
                x.shape[1], -1, -1, -1), stride=2, groups=x.shape[1])
            x = layer(torch.cat([x[..., :sk.shape[2], :sk.shape[3]], sk], 1))
        return x


class D3Net(nn.Module):
    def __init__(self,
                 freq_split_idx: int,
                 low_specs: Tuple[List[Dict], List[Dict]],
                 hi_specs: Tuple[List[Dict], List[Dict]],
                 full_specs: Tuple[List[Dict], List[Dict]],
                 last_n_layers=1):
        super().__init__()

        self.freq_split_idx = freq_split_idx

        for spec_list in low_specs + hi_specs + full_specs:
            for spec in spec_list:
                spec['last_n_layers'] = last_n_layers

        self.full = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            UNet(32, full_specs[0], full_specs[1])
        )
        unet = UNet(32, low_specs[0], low_specs[1])
        self.low = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            unet,
        )

        unet = UNet(8, hi_specs[0], hi_specs[1])
        self.high = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            unet,
            nn.Conv2d(unet.get_output_channels(),
                      self.low[1].get_output_channels(), 1, bias=False)
        )

        in_channels = self.full[1].get_output_channels(
        ) + self.low[1].get_output_channels()

        d2 = D2_block(in_channels, 12, 3, last_n_layers=3)
        out_channels = d2.get_output_channels()
        self.final = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            d2,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, 3, padding=1)
        )

    def forward(self, spec):
        low_spec, high_spec = spec[:, :,
                                   :self.freq_split_idx], spec[:, :, self.freq_split_idx:1600]
        full_spec = spec[:, :, :1600]
        low = self.low(low_spec)
        high = self.high(high_spec)
        full = self.full(full_spec)

        hi_low = torch.cat([low, high], 2)
        final = torch.cat([hi_low, full], 1)
        return F.pad(self.final(final), [0, 0, 0, spec.shape[2] - 1600], value=-15)


def get_vocals_model(last_n_layers=1):
    return D3Net(
        256,
        (
            [
                {'k': 16, 'L': 5, 'M': 2},
                {'k': 18, 'L': 5, 'M': 2},
                {'k': 20, 'L': 5, 'M': 2},
                {'k': 22, 'L': 5, 'M': 2},
            ],
            [
                {'k': 20, 'L': 4, 'M': 2},
                {'k': 18, 'L': 4, 'M': 2},
                {'k': 16, 'L': 4, 'M': 2}
            ]
        ),
        (
            [
                {'k': 2, 'L': 1, 'M': 1},
                {'k': 2, 'L': 1, 'M': 1},
                {'k': 2, 'L': 1, 'M': 1},
                {'k': 2, 'L': 1, 'M': 1}
            ],
            [
                {'k': 2, 'L': 1, 'M': 1},
                {'k': 2, 'L': 1, 'M': 1},
                {'k': 2, 'L': 1, 'M': 1}
            ]
        ),
        (
            [
                {'k': 13, 'L': 4, 'M': 2},
                {'k': 14, 'L': 5, 'M': 2},
                {'k': 15, 'L': 6, 'M': 2},
                {'k': 16, 'L': 7, 'M': 2},
                {'k': 17, 'L': 8, 'M': 2},
            ],
            [
                {'k': 14, 'L': 6, 'M': 2},
                {'k': 13, 'L': 5, 'M': 2},
                {'k': 12, 'L': 4, 'M': 2},
                {'k': 11, 'L': 4, 'M': 2}
            ]
        ),
        last_n_layers
    )


if __name__ == "__main__":
    m = get_vocals_model().cuda()  # .half()
    # print(m)
    # torch.save(m, 'model_size_test.pth')
    x = torch.rand(1, 2, 2049, 256).cuda()  # .half()
    y = m(x)
    print(y.shape)
    exit(0)
    # m = D2_block(32, 6, 4, 4)
    # m = D3_block(32, 2, k=13, L=5)
    m = UNet(
        32,
        [
            {'k': 16, 'L': 5, 'M': 2},
            {'k': 18, 'L': 5, 'M': 2},
            {'k': 20, 'L': 5, 'M': 2},
            {'k': 22, 'L': 5, 'M': 2},
        ],
        [
            {'k': 20, 'L': 4, 'M': 2},
            {'k': 18, 'L': 4, 'M': 2},
            {'k': 16, 'L': 4, 'M': 2}
        ]
    ).cuda()
    print(m)
    print(m.get_output_channels())
    print(m)
    print(m.get_output_channels())
