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
