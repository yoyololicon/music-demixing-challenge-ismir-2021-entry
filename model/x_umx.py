import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import quantize, get_code_usage


class X_UMX(nn.Module):
    __constants__ = ['max_bins']

    max_bins: int

    def __init__(
        self,
        n_fft=4096,
        hidden_channels=512,
        max_bins=None,
        nb_channels=2,
        nb_layers=3,
        pre_avg=True,
        post_avg=True,
        vq_position=None,
        latent_dim=256,
        n_code=2048
    ):
        super().__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bins:
            self.max_bins = max_bins
        else:
            self.max_bins = self.nb_output_bins
        self.hidden_channels = hidden_channels
        self.n_fft = n_fft
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.pre_avg = pre_avg
        self.post_avg = post_avg

        if not pre_avg:
            raise NotImplementedError
        else:
            n_pre_in = 1  # four feature maps are averaged

        self.vq_position = vq_position

        self.input_means = nn.Parameter(torch.zeros(4 * self.max_bins))
        self.input_scale = nn.Parameter(torch.ones(4 * self.max_bins))

        self.output_means = nn.Parameter(torch.zeros(4 * self.nb_output_bins))
        self.output_scale = nn.Parameter(torch.ones(4 * self.nb_output_bins))

        self.affine1 = nn.Sequential(
            nn.Conv1d(nb_channels * self.max_bins * 4,
                      hidden_channels * 4, 1, bias=False, groups=4),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.Tanh()
        )
        self.bass_lstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.drums_lstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.vocals_lstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)
        self.other_lstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels // 2,
            num_layers=nb_layers,
            dropout=0.4,
            bidirectional=True)

        if not post_avg:
            # four sources and skip connected inputs
            n_post_in = 4 + n_pre_in
        else:
            # averaged output feature maps and skip connected inputs
            n_post_in = 1 + n_pre_in  

        self.affine2 = nn.Sequential(
            nn.Conv1d(hidden_channels * n_post_in,
                      hidden_channels * 4, 1, bias=False),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels * 4,
                      nb_channels * self.nb_output_bins * 4, 1, bias=False, groups=4),
            nn.BatchNorm1d(nb_channels * self.nb_output_bins * 4),
        )

        if vq_position:
            assert vq_position in ['pre_lstm', 'post_lstm']
            self.codebook = nn.Embedding(n_code, latent_dim)
            nn.init.xavier_uniform_(self.codebook.weight) 
            
        if vq_position == 'pre_lstm':
            pre_q_dim = hidden_channels
        else:
            pre_q_dim = hidden_channels * n_post_in

        if latent_dim != pre_q_dim:
            self.projection = nn.Linear(pre_q_dim, latent_dim)
            self.de_projection = nn.Linear(latent_dim, pre_q_dim)
        else:
            self.projection = nn.Identity()
            self.de_projection = nn.Identity()

    def forward(self, spec: torch.Tensor):
        batch, channels, bins, frames = spec.shape
        spec = spec[..., :self.max_bins, :]

        x = (spec.unsqueeze(1) + self.input_means.view(4, 1, -1, 1)) * \
            self.input_scale.view(4, 1, -1, 1)

        x = x.view(batch, -1, frames)  # same shape as cross_2
        cross_1 = self.affine1(x).view(batch, 4, -1, frames).mean(1)

        if self.vq_position == 'pre_lstm':
            e = self.projection(cross_1.view(batch, frames, -1))
            q, indices = quantize(e, self.codebook.weight)
            code_usage = get_code_usage(self.codebook, indices)
            e_post_q = self.de_projection(q)
            cross_1 = e_post_q.view(batch, -1, frames)

        cross_1 = cross_1.permute(2, 0, 1)
        bass, *_ = self.bass_lstm(cross_1)
        drums, *_ = self.drums_lstm(cross_1)
        others, *_ = self.other_lstm(cross_1)
        vocals, *_ = self.vocals_lstm(cross_1)

        if not self.post_avg:
            post_in = torch.cat([bass, drums, vocals, others], 2)
        else:
            post_in = (bass + drums + vocals + others) * 0.25
        cross_2 = torch.cat([cross_1, post_in], 2).permute(1, 2, 0)

        if self.vq_position == 'post_lstm':
            e = self.projection(cross_2.reshape(batch, frames, -1))
            q, indices = quantize(e, self.codebook.weight)
            code_usage = get_code_usage(self.codebook, indices)
            e_post_q = self.de_projection(q)
            cross_2 = e_post_q.view(batch, -1, frames)

        mask = self.affine2(cross_2).view(batch, 4, channels, bins, frames) * \
            self.output_scale.view(4, 1, -1, 1) + \
            self.output_means.view(4, 1, -1, 1)
        return mask.relu()


if __name__ == "__main__":
    net = X_UMX(max_bins=2000)
    net_vq = X_UMX(max_bins=2000, vq_position='pre_lstm')
    net_no_post_avg = X_UMX(max_bins=2000, post_avg=False)
    net_vq_no_post_avg = X_UMX(
        max_bins=2000, post_avg=False, vq_position='post_lstm'
    )

    spec = torch.rand(1, 2, 2049, 10)

    y = net(spec)
    print(y.shape)

    y_vq = net_vq(spec)
    print(y_vq.shape)

    y_npa = net_no_post_avg(spec)
    print(y_npa.shape)

    y_vq_npa = net_vq_no_post_avg(spec)
    print(y_vq_npa.shape)

    # x = torch.randn(8, 44100)
    # print(net.t2f(x).shape)
