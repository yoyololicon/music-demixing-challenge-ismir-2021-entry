import torch
from torch import nn
import torch.nn.functional as F


class MultiUNet(nn.Module):
    def __init__(self,
                 n_targets=4,
                 max_bins=1600):
        super().__init__()
        assert n_targets > 1
        self.max_bins = max_bins
        self.n_targets = n_targets
        self.conv_n_filters = [16, 32, 64, 128, 256, 512]

        in_channels = 2
        self.down_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              self.conv_n_filters[0] * n_targets,
                              5,
                              stride=2,
                              padding=2),
                    nn.BatchNorm2d(self.conv_n_filters[0] * n_targets),
                    nn.ELU()
                )
            ]
        )
        in_channels = self.conv_n_filters[0]

        for out_channels in self.conv_n_filters[1:-1]:
            self.down_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels * n_targets,
                              out_channels * n_targets,
                              5,
                              stride=2,
                              padding=2,
                              groups=n_targets),
                    nn.BatchNorm2d(out_channels * n_targets),
                    nn.ELU()
                )
            )
            in_channels = out_channels
        self.down_convs.append(
            nn.Conv2d(in_channels * n_targets,
                      self.conv_n_filters[-1] * n_targets,
                      5,
                      stride=2,
                      padding=2,
                      groups=n_targets)
        )

        self.up_convs = nn.ModuleList()
        self.skip_join = nn.ModuleList()
        concat_channels = 0
        drop_count = 3
        for i in range(len(self.conv_n_filters) - 2, -1, -1):
            layers = [
                nn.BatchNorm2d(self.conv_n_filters[i + 1] * n_targets),
                nn.ELU()
            ]
            if drop_count:
                layers += [nn.Dropout2d(0.5, inplace=True)]

            layers.append(
                nn.ConvTranspose2d(
                    self.conv_n_filters[i + 1] * n_targets,
                    self.conv_n_filters[i] * n_targets,
                    5,
                    stride=2,
                    padding=2,
                    groups=n_targets
                )
            )
            self.up_convs.append(
                nn.Sequential(*layers)
            )
            if concat_channels:
                layers = [
                    nn.ConvTranspose2d(
                        self.conv_n_filters[i + 1],
                        self.conv_n_filters[i] * n_targets,
                        5,
                        stride=2,
                        padding=2,
                        bias=False
                    )
                ]
                if drop_count:
                    layers = [nn.Dropout2d(0.5, inplace=True)] + layers
                self.skip_join.append(
                    nn.Sequential(*layers) if len(layers) > 1 else layers[0]
                )
            drop_count -= 1
            concat_channels = self.conv_n_filters[i]

        self.end1 = nn.Sequential(
            nn.BatchNorm2d(concat_channels * n_targets),
            nn.ELU(),
            nn.ConvTranspose2d(concat_channels * n_targets,
                               concat_channels * n_targets, 5, 2, 2, groups=n_targets)
        )
        self.skip_join.append(
            nn.ConvTranspose2d(concat_channels,
                               concat_channels * n_targets, 5, 2, 2, bias=False)
        )
        self.end2 = nn.Sequential(
            nn.BatchNorm2d(concat_channels * n_targets),
            nn.ELU(),
            nn.Conv2d(concat_channels * n_targets, 2 * n_targets,
                      3, padding=2, dilation=2, groups=n_targets)
        )

    def forward(self, spec):
        x = spec[:, :, :self.max_bins].add(1e-8).log_()

        skips = []
        for conv in self.down_convs:
            x = conv(x)
            skips.append(x)
        skips.pop()

        x = self.up_convs[0](x)

        for conv, sk, sk_join in zip(self.up_convs[1:], skips[:0:-1], self.skip_join[:-1]):
            sk = sk.view(sk.shape[0], self.n_targets, -1,
                         sk.shape[2], sk.shape[3]).mean(1)
            if x.shape[2] < sk.shape[2] or x.shape[3] < sk.shape[3]:
                x = F.pad(x, [0, sk.shape[3] - x.shape[3],
                              0, sk.shape[2] - x.shape[2]])
            sk = sk_join(sk)
            x = conv(x)
            x += sk

        sk = skips[0]
        sk = sk.view(sk.shape[0], self.n_targets, -1,
                     sk.shape[2], sk.shape[3]).mean(1)
        if x.shape[2] < sk.shape[2] or x.shape[3] < sk.shape[3]:
            x = F.pad(x, [0, sk.shape[3] - x.shape[3],
                          0, sk.shape[2] - x.shape[2]])
        x = self.end1(x) + self.skip_join[-1](sk)
        x = self.end2(x)
        x = x.view(x.shape[0], self.n_targets, -1, x.shape[2], x.shape[3])
        x = F.softmax(x, 1)
        return F.pad(x, [0, spec.shape[3] - x.shape[4], 0, spec.shape[2] - x.shape[3]], value=1 / self.n_targets)


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
