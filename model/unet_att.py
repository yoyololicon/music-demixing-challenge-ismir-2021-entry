import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from .d3net import D3_block


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.4):
        super().__init__()

        self.convs = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
        )

        if stride != 1:
            setattr(self, "skip", nn.Conv2d(
                in_channels, out_channels, 1, stride=stride))
        else:
            setattr(self, "skip", None)

    def forward(self, x):
        out = self.convs(x)
        if self.skip is not None:
            out += self.skip(x)
        return out


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.4):
        super().__init__()

        self.convs = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size // 2),
        )

        if stride != 1:
            setattr(self, "skip", nn.ConvTranspose2d(
                in_channels, out_channels, 1, stride=stride))
        else:
            setattr(self, "skip", None)

    def forward(self, x):
        out = self.convs(x)
        if self.skip is not None:
            out += self.skip(x)
        return out


class MultiHeadAttention(nn.Module):
    __constants__ = [
        'out_channels',
        'd_model',
        'n_heads',
        'query_shape',
        'memory_flange'
    ]

    max_bins: int
    d_model: int
    n_heads: int
    query_shape: int
    memory_flange: int

    def __init__(self, in_channels, out_channels, d_model=32, n_heads=8, query_shape=24, memory_flange=8):
        super().__init__()

        self.out_channels = out_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_shape = query_shape
        self.memory_flange = memory_flange

        self.qkv_conv = nn.Conv2d(in_channels, d_model * 3, 3, padding=1)
        self.out_conv = nn.Conv2d(
            d_model, out_channels, 3, padding=1, bias=False)

    def _pad_to_multiple_2d(self, x: torch.Tensor, query_shape: int):
        t = x.shape[-1]
        offset = t % query_shape
        if offset != 0:
            offset = query_shape - offset

        if offset > 0:
            return F.pad(x, [0, offset])
        return x

    def forward(self, x):
        qkv = self._pad_to_multiple_2d(self.qkv_conv(x), self.query_shape)
        qkv = qkv.view((qkv.shape[0], self.n_heads, -1) + qkv.shape[2:])
        q, k, v = qkv.chunk(3, 2)

        k_depth_per_head = self.d_model // self.n_heads
        q = q * k_depth_per_head ** -0.5

        k = F.pad(k, [self.memory_flange] * 2)
        v = F.pad(v, [self.memory_flange] * 2)

        unfold_q = q.reshape(
            q.shape[:4] + (q.shape[4] // self.query_shape, self.query_shape))
        unfold_k = k.unfold(-1, self.query_shape +
                            self.memory_flange * 2, self.query_shape)
        unfold_v = v.unfold(-1, self.query_shape +
                            self.memory_flange * 2, self.query_shape)

        unfold_q = unfold_q.permute(0, 1, 4, 3, 5, 2)
        tmp = unfold_q.shape
        unfold_q = unfold_q.reshape(
            -1, unfold_q.shape[-2] * unfold_q.shape[-3], k_depth_per_head)
        unfold_k = unfold_k.permute(0, 1, 4, 2, 3, 5).reshape(
            unfold_q.shape[0], k_depth_per_head, -1)
        unfold_v = unfold_v.permute(0, 1, 4, 3, 5, 2).reshape(
            unfold_q.shape[0], -1, k_depth_per_head)

        bias = (unfold_k.abs().sum(-2, keepdim=True)
                == 0).to(unfold_k.dtype) * -1e-9
        # correct value should be -1e9, we type this by accident and use it during the whole competition
        # so just leave it what it was :)

        logits = unfold_q @ unfold_k + bias
        weights = logits.softmax(-1)
        out = weights @ unfold_v

        out = out.view(tmp).permute(0, 1, 5, 3, 2, 4)
        out = out.reshape(out.shape[0], out.shape[1] *
                          out.shape[2], out.shape[3], -1)
        out = out[..., :x.shape[2], :x.shape[3]]

        return self.out_conv(out)


class UNetAttn(nn.Module):
    def __init__(self, max_bins=1487, k=12):
        super().__init__()
        self.max_bins = max_bins
        layers = 6
        self.conv_n_filters = []

        in_channels = 2
        self.down_convs = nn.ModuleList()

        for i in range(layers):
            L = max(2, layers - i - 1)
            tmp = D3_block(
                in_channels, M=2,
                k=k, L=L, last_n_layers=L
            )
            out_channels = tmp.get_output_channels()
            self.down_convs.append(
                nn.Sequential(
                    tmp,
                    nn.AvgPool2d(2, 2)
                )
            )
            in_channels = out_channels
            self.conv_n_filters.append(out_channels)

        self.attn = nn.Sequential(
            MultiHeadAttention(in_channels, 256, 128,
                               query_shape=2, memory_flange=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            MultiHeadAttention(256, 256, 128,
                               query_shape=2, memory_flange=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up_convs = nn.ModuleList()
        concat_channels = 256
        n_droupout = 3
        for i in range(len(self.conv_n_filters) - 2, -1, -1):
            L = max(2, layers - i - 1)
            d3 = D3_block(
                self.conv_n_filters[i + 1] + concat_channels, M=2,
                k=k, L=L, last_n_layers=L
            )
            tmp = [
                d3,
                nn.ConvTranspose2d(d3.get_output_channels(
                ), self.conv_n_filters[i], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.conv_n_filters[i]),
                nn.ReLU(inplace=True)
            ]
            if n_droupout > 0:
                tmp.append(nn.Dropout2d(0.4))
                n_droupout -= 1
            self.up_convs.append(
                nn.Sequential(
                    *tmp
                )
            )
            concat_channels = self.conv_n_filters[i]

        self.end = nn.Sequential(
            nn.ConvTranspose2d(concat_channels * 2,
                               concat_channels, 5, 2, 2, bias=False),
            nn.BatchNorm2d(concat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(concat_channels, 8, 3, padding=2, dilation=2)
        )

    def forward(self, spec):
        x = spec[:, :, :self.max_bins]

        skips = []
        for conv in self.down_convs:
            x = conv(x)
            skips.append(x)
        skips.pop()

        x = torch.cat([x, self.attn(x)], 1)

        i = len(skips) - 1
        for conv in self.up_convs:
            sk = skips[i]
            x = conv(x)
            if x.shape[2] < sk.shape[2] or x.shape[3] < sk.shape[3]:
                x = F.pad(x, [0, sk.shape[3] - x.shape[3],
                              0, sk.shape[2] - x.shape[2]])
            x = torch.cat([x, sk], 1)
            i -= 1
        x = self.end(x)
        x = x.sigmoid()

        x = F.pad(x, [0, spec.shape[3] - x.shape[3], 0,
                      spec.shape[2] - x.shape[2]], value=0.25)
        x = x.view(x.shape[0], 4, 2, x.shape[2], x.shape[3])
        return x


if __name__ == "__main__":
    net = UNetAttn()
    # net = torch.jit.script(net)
    print(net)
    print(sum(p.numel() for p in net.parameters()
              if p.requires_grad), net.conv_n_filters)
    x = torch.rand(1, 2, 2049, 512)
    y = net(x)
    print(y.shape)
