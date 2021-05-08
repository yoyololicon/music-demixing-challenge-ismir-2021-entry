import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class UNet(nn.Module):
    def __init__(self, max_bins=1600):
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

        self.up_convs = nn.ModuleList()
        concat_channels = 0
        for i in range(len(self.conv_n_filters) - 2, -1, -1):
            self.up_convs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.conv_n_filters[i + 1] + concat_channels,
                        self.conv_n_filters[i],
                        5,
                        stride=2,
                        padding=2
                    ),
                    nn.ELU(),
                    nn.BatchNorm2d(self.conv_n_filters[i]),
                    nn.Dropout2d(0.5, inplace=True)
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

        for conv, sk in zip(self.up_convs, skips[::-1]):
            x = conv(x)
            if x.shape[2] < sk.shape[2] or x.shape[3] < sk.shape[3]:
                x = F.pad(x, [0, sk.shape[3] - x.shape[3],
                              0, sk.shape[2] - x.shape[2]])
            x = torch.cat([x, sk], 1)
        x = self.end(x)

        return F.pad(x, [0, spec.shape[3] - x.shape[3], 0, spec.shape[2] - x.shape[2]])
