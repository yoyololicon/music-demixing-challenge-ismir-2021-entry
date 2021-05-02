import torch
from torch import nn


class _Base(nn.Module):
    def get_output_channels(self):
        raise NotImplementedError


class D2_block(_Base):
    def __init__(self,
                 in_channels,
                 k,
                 L,
                 last_n_layer=3):
        super().__init__()

        self.in_channels = in_channels
        self.k = k
        self.L = L
        self.last_N = last_n_layer

        self.conv_layers = nn.ModuleList()

        self.output_sizes = [k * (i + 1) for i in range(L)]
        self.input_sizes = [in_channels] + [k * (i + 1) for i in range(L - 1)]
        for i in range(L):
            if i:
                self.conv_layers.append(
                    nn.Sequential(
                        nn.BatchNorm2d(self.input_sizes[i]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            self.input_sizes[i],
                            sum(self.output_sizes[i:]),
                            3,
                            padding=2 ** i,
                            dilation=2 ** i,
                            bias=False
                        )
                    )
                )
            else:
                self.conv_layers.append(
                    nn.Conv2d(
                        self.input_sizes[i],
                        sum(self.output_sizes[i:]),
                        3,
                        padding=2 ** i,
                        dilation=2 ** i,
                        bias=False
                    )
                )

    def get_output_channels(self):
        return sum(self.output_sizes[-self.last_N:])

    def forward(self, input: torch.Tensor):
        # the input should be already BN + ReLU before
        x = self.conv_layers[0](input)
        input, *skips = x.split(self.output_sizes, 1)

        outputs = [input]
        for i in range(1, self.L):
            input, * \
                tmp = self.conv_layers[i](input).split(
                    self.output_sizes[i:], 1)
            outputs.append(input)
            input = input + skips.pop(0)
            skips = [s + t for s, t in zip(skips, tmp)]

        assert len(skips) == 0
        return torch.cat(outputs[-self.last_N:], 1)


class D3_block(_Base):
    def __init__(self,
                 in_channels,
                 M,
                 *args,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels

        self.bn_layers = nn.ModuleList()
        self.d2_layers = nn.ModuleList()

        concat_channels = in_channels
        for i in range(M):
            self.bn_layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(concat_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.d2_layers.append(
                D2_block(in_channels, *args, **kwargs)
            )
            concat_channels = self.d2_layers[-1].get_output_channels()
            in_channels += concat_channels

    def get_output_channels(self):
        return self.in_channels + sum(l.get_output_channels() for l in self.d2_layers)

    def forward(self, input):
        raw_inputs = [input]
        bn_inputs = []
        for bn, d2 in zip(self.bn_layers, self.d2_layers):
            bn_input = bn(input)
            bn_inputs.append(bn_input)
            input = d2(torch.cat(bn_inputs, 1))
            raw_inputs.append(input)
        return torch.cat(raw_inputs, 1)


if __name__ == "__main__":
    m = D3_block(
        32, 2, 17, 8, last_n_layer=3
    )

    print(m.get_output_channels())
    print(m)

    x = torch.rand(1, 32, 256, 256)
    y = m(x)
    print(y.shape)
