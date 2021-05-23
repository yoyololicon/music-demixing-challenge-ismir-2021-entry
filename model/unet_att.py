import torch
from torch import nn
import torch.nn.functional as F
import math

from torch.nn.modules.dropout import Dropout, Dropout2d


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
    def __init__(self, in_channels, out_channels, d_model=32, n_heads=8, query_shape=(128, 24), memory_flange=(8, 8)):
        super().__init__()

        self.out_channels = out_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_shape = query_shape
        self.memory_flange = memory_flange

        self.q_conv = nn.Conv2d(in_channels, d_model, 3, padding=1)
        self.k_conv = nn.Conv2d(in_channels, d_model, 3, padding=1)
        self.v_conv = nn.Conv2d(in_channels, d_model, 3, padding=1)
        self.out_conv = nn.Conv2d(
            d_model, out_channels, 3, padding=1, bias=False)

    def _pad_to_multiple_2d(self, x: torch.Tensor, query_shape: tuple):
        shape = x.shape[-2:]
        offsets = [shape[0] % query_shape[0], shape[1] % query_shape[1]]
        for i in range(2):
            if offsets[i] != 0:
                offsets[i] = query_shape[i] - offsets[i]

        if sum(offsets) > 0:
            return F.pad(x, [0, offsets[1], 0, offsets[0]])
        return x

    def forward(self, x):
        q = self._pad_to_multiple_2d(self.q_conv(x), self.query_shape)
        k = self._pad_to_multiple_2d(self.k_conv(x), self.query_shape)
        v = self._pad_to_multiple_2d(self.v_conv(x), self.query_shape)

        q = q.view(q.shape[0], self.n_heads, -1, *q.shape[2:])
        k = k.view(k.shape[0], self.n_heads, -1, *k.shape[2:])
        v = v.view(v.shape[0], self.n_heads, -1, *v.shape[2:])

        k_depth_per_head = self.d_model // self.n_heads
        q *= k_depth_per_head ** -0.5

        k = F.pad(k, [self.memory_flange[1]] * 2 + [self.memory_flange[0]] * 2)
        v = F.pad(v, [self.memory_flange[1]] * 2 + [self.memory_flange[0]] * 2)

        unfold_q = q.view(*q.shape[:3],
                          q.shape[3] // self.query_shape[0], self.query_shape[0],
                          q.shape[4] // self.query_shape[1], self.query_shape[1])
        new_shape = unfold_q.shape[:3] + (
            unfold_q.shape[3], self.memory_flange[0] * 2 + unfold_q.shape[4],
            unfold_q.shape[5], self.memory_flange[1] * 2 + unfold_q.shape[6])
        new_stride = unfold_q.stride()
        unfold_k = torch.as_strided(k, new_shape, new_stride)
        unfold_v = torch.as_strided(v, new_shape, new_stride)

        unfold_q = unfold_q.permute(0, 1, 3, 5, 4, 6, 2)
        tmp = unfold_q.shape
        unfold_q = unfold_q.reshape(
            -1, self.query_shape[0] * self.query_shape[1], k_depth_per_head)
        unfold_k = unfold_k.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            unfold_q.shape[0], k_depth_per_head, -1)
        unfold_v = unfold_v.permute(0, 1, 3, 5, 4, 6, 2).reshape(
            unfold_q.shape[0], -1, k_depth_per_head)

        logits = unfold_q @ unfold_k
        weights = logits.softmax(-1)
        out = weights @ unfold_v

        out = out.view(*tmp).permute(0, 1, 6, 2, 4, 3, 5)
        out = out.reshape(out.shape[0], out.shape[1] *
                          out.shape[2], out.shape[3] * out.shape[4], -1)
        out = out[..., :x.shape[2], :x.shape[3]]

        return self.out_conv(out)


class UNetAttn(nn.Module):
    def __init__(self, max_bins=1487):
        super().__init__()
        self.max_bins = max_bins
        self.en = nn.Conv2d(2, 32, 7, padding=3)

        self.en_l1 = nn.Sequential(
            ConvBlock(32, 32, 3, 2),
            ConvBlock(32, 32, 3)
        )
        self.en_l2 = nn.Sequential(
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 64, 3),
            ConvBlock(64, 64, 3)
        )
        self.en_l3 = nn.Sequential(
            ConvBlock(64, 128, 3, 2),
            ConvBlock(128, 128, 3),
            ConvBlock(128, 128, 3),
            ConvBlock(128, 128, 3)
        )
        self.en_l4 = nn.Sequential(
            ConvBlock(128, 256, 3, 2),
            ConvBlock(256, 256, 3),
            ConvBlock(256, 256, 3),
            ConvBlock(256, 256, 3),
            ConvBlock(256, 256, 3)
        )

        self.f_attn = nn.Sequential(
            MultiHeadAttention(256, 64, query_shape=(100, 32)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            MultiHeadAttention(64, 128, d_model=64, query_shape=(64, 16))
        )

        self.dec = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(128 + 256),
                    nn.Conv2d(256 + 128, 256, 1)
                ),
                ConvTransposeBlock(256, 128, 3, stride=2)
            ]),
            nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(256),
                    nn.Dropout2d(0.4),
                    nn.Conv2d(256, 128, 1)
                ),
                ConvTransposeBlock(128, 64, 3, stride=2)
            ]),
            nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(128),
                    nn.Dropout2d(0.4),
                    nn.Conv2d(128, 64, 1)
                ),
                ConvTransposeBlock(64, 64, 3, stride=2)
            ]),
            nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(64 + 32),
                    nn.Dropout2d(0.4),
                    nn.Conv2d(64 + 32, 64, 1)
                ),
                ConvTransposeBlock(64, 64, 3, stride=2)
            ]),
        ])

        self.end = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 8, 1)
        )

    def forward(self, spec):
        x = spec[..., :self.max_bins, :]

        x = self.en(x)
        x = self.en_l1(x)
        skips = [x]
        x = self.en_l2(x)
        skips.append(x)
        x = self.en_l3(x)
        skips.append(x)
        x = self.en_l4(x)
        skips.append(self.f_attn(x))
        for layers, sk in zip(self.dec, skips[::-1]):
            conv, tconv = layers
            if x.shape[2] < sk.shape[2] or x.shape[3] < sk.shape[3]:
                x = F.pad(x, [0, sk.shape[3] - x.shape[3],
                              0, sk.shape[2] - x.shape[2]])
            res = x
            x = torch.cat([x, sk], 1)
            x = conv(x) + res
            x = tconv(x)

        x = self.end(x)
        x = x.view(x.shape[0], 4, 2, *x.shape[2:]).sigmoid()
        x = F.pad(x, [0, spec.shape[3] - x.shape[4], 0,
                      spec.shape[2] - x.shape[3]], value=0.25)
        return x


if __name__ == "__main__":
    net = UNetAttn()
    #net = torch.jit.script(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.rand(1, 2, 2049, 256)
    y = net(x)
    print(y.shape)
