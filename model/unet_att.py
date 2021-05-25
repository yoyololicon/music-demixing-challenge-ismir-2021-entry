import torch
from torch import nn
import torch.nn.functional as F
import math


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
    def __init__(self, in_channels, out_channels, d_model=32, n_heads=8, query_shape=24, memory_flange=8):
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

    def _pad_to_multiple_2d(self, x: torch.Tensor, query_shape: int):
        t = x.shape[-1]
        offset = t % query_shape
        if offset != 0:
            offset = query_shape - offset

        if offset > 0:
            return F.pad(x, [0, offset])
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

        k = F.pad(k, [self.memory_flange] * 2)
        v = F.pad(v, [self.memory_flange] * 2)

        unfold_q = q.view(*q.shape[:4],
                          q.shape[4] // self.query_shape, self.query_shape)
        new_shape = unfold_q.shape[:4] + (
            unfold_q.shape[4], self.memory_flange * 2 + unfold_q.shape[5])
        new_stride = unfold_q.stride()
        unfold_k = torch.as_strided(k, new_shape, new_stride)
        unfold_v = torch.as_strided(v, new_shape, new_stride)

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

        logits = unfold_q @ unfold_k + bias
        weights = logits.softmax(-1)
        out = weights @ unfold_v

        out = out.view(*tmp).permute(0, 1, 5, 3, 2, 4)
        out = out.reshape(out.shape[0], out.shape[1] *
                          out.shape[2], out.shape[3], -1)
        out = out[..., :x.shape[2], :x.shape[3]]

        return self.out_conv(out)


class UNetAttn(nn.Module):
    def __init__(self, max_bins=1487):
        super().__init__()
        self.max_bins = max_bins
        self.conv_n_filters = [16, 32, 64, 128, 256, 512]

        in_channels = 2
        self.down_convs = nn.ModuleList()

        for out_channels in self.conv_n_filters[:-1]:
            self.down_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels,
                              5, stride=2, padding=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU()
                )
            )
            in_channels = out_channels

        self.down_convs.append(
            nn.Conv2d(
                in_channels, self.conv_n_filters[-1], 5, stride=2, padding=2)
        )
        in_channels = self.conv_n_filters[-1]

        self.attn = nn.Sequential(
            MultiHeadAttention(in_channels, 128, 64,
                               query_shape=2, memory_flange=3),
            nn.BatchNorm2d(128),
            nn.ELU(),
            MultiHeadAttention(128, 256, 128,
                               query_shape=2, memory_flange=3),
            nn.BatchNorm2d(256),
            nn.ELU()
        )

        self.up_convs = nn.ModuleList()
        concat_channels = 256
        n_droupout = 3
        for i in range(len(self.conv_n_filters) - 2, -1, -1):
            tmp = [
                nn.ConvTranspose2d(
                    self.conv_n_filters[i + 1] + concat_channels,
                    self.conv_n_filters[i],
                    5,
                    stride=2,
                    padding=2
                ),
                nn.ELU(),
                nn.BatchNorm2d(self.conv_n_filters[i])
            ]
            if n_droupout > 0:
                tmp.append(nn.Dropout2d(0.5, inplace=True))
                n_droupout -= 1
            self.up_convs.append(
                nn.Sequential(
                    *tmp
                )
            )
            concat_channels = self.conv_n_filters[i]

        self.end = nn.Sequential(
            nn.ConvTranspose2d(concat_channels * 2, concat_channels, 5, 2, 2),
            nn.ELU(),
            nn.BatchNorm2d(concat_channels),
            nn.Conv2d(concat_channels, 2, 3, padding=2, dilation=2),
            nn.Sigmoid()
        )

    def forward(self, spec):
        x = spec[:, :, :self.max_bins]

        skips = []
        for conv in self.down_convs:
            x = conv(x)
            skips.append(x)
        skips.pop()

        x = torch.cat([x, self.attn(x)], 1)

        for conv, sk in zip(self.up_convs, skips[::-1]):
            x = conv(x)
            if x.shape[2] < sk.shape[2] or x.shape[3] < sk.shape[3]:
                x = F.pad(x, [0, sk.shape[3] - x.shape[3],
                              0, sk.shape[2] - x.shape[2]])
            x = torch.cat([x, sk], 1)
        x = self.end(x)

        x = F.pad(x, [0, spec.shape[3] - x.shape[3], 0,
                      spec.shape[2] - x.shape[2]], value=0.25)
        return x


if __name__ == "__main__":
    net = UNetAttn()
    #net = torch.jit.script(net)
    print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.rand(1, 2, 2049, 512)
    y = net(x)
    print(y.shape)
