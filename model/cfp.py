"""
adopted code from
https://github.com/yoyololicon/HPNN-multipitch-estimation/blob/main/modules.py
"""
import torch
import numpy as np
from torch import nn, fft
import torch.nn.functional as F




class MLC(nn.Module):
    def __init__(self, sr, n_fft, hop_size,
                 g=[0.24, 0.6, 1],
                 hipass_f=27.5, lowpass_t=0.24):
        super().__init__()
        self.hop_size = hop_size
        self.n_fft = n_fft

        self.hpi = int(hipass_f * n_fft / sr) + 1
        self.lpi = int(lowpass_t * sr / 1000) + 1
        self.g = g

        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x):
        x = torch.stft(x, self.n_fft, self.hop_size, window=self.window, center=False, normalized=True,
                       onesided=False, return_complex=True).abs().pow(2)
        spec = x ** self.g[0]

        ceps = torch.zeros_like(spec)
        for d, g in enumerate(self.g[1:]):
            if d % 2:
                spec = fft.fft(ceps, dim=-2, norm="ortho").real
                spec[:, :self.hpi, :] = spec[:, -self.hpi:, :] = 0
                spec = spec.relu_() ** g
            else:
                ceps = fft.fft(spec, dim=-2, norm="ortho").real
                ceps[:, :self.lpi, :] = ceps[:, -self.lpi:, :] = 0
                ceps = ceps.relu_() ** g
        return ceps, spec


class CFP(nn.Module):
    def __init__(self, in_channels, sr, start_midi=21, end_midi=108, division=1, norm=False):
        """
        Parameters
        ----------
        in_channels: int
            window size
        sr: int
            sample rate
        harms_range: int
            The extended area above (or below) the piano pitch range (in semitones)
            25 : though somewhat larger, to ensure the coverage is large enough (if division=1, 24 is sufficient)
        division: int
            The division number for filterbank frequency resolution. The frequency resolution is 1 / division (semitone)
        norm: bool
            If set to True, normalize each filterbank so the weight of each filterbank sum to 1.
        """
        super().__init__()
        step = 1 / division
        # midi_num shape = (88 + harms_range) * division + 2
        # this implementation make sure if we group midi_num with a size of division
        # each group will center at the piano pitch number and the extra pitch range
        # E.g., division = 2, midi_num = [20.25, 20.75, 21.25, ....]
        #       dividion = 3, midi_num = [20.33, 20.67, 21, 21.33, ...]
        midi_num = np.arange(start_midi - 0.5 - step / 2,
                             end_midi + 0.5 + step, step)
        # self.midi_num = midi_num

        fd = 440 * np.power(2, (midi_num - 69) / 12)
        # self.fd = fd

        self.effected_dim = in_channels // 2 + 1
        # // 2 : the spectrum/ cepstrum are symmetric

        x = np.arange(self.effected_dim)
        freq_f = x * sr / in_channels
        freq_t = sr / x[1:]
        # avoid explosion; x[0] is always 0 for cepstrum

        inter_value = np.array([0, 1., 0])
        idxs = np.digitize(freq_f, fd)

        num_filters = (end_midi - start_midi + 1) * division
        self.register_buffer('filter_f_weight', torch.zeros(
            num_filters, self.effected_dim))
        self.register_buffer('filter_t_weight', torch.zeros(
            num_filters, self.effected_dim))

        cols, rows, values = [], [], []
        for i in range(num_filters):
            idx = np.where((idxs == i + 1) | (idxs == i + 2))[0]
            c = idx
            r = np.broadcast_to(i, idx.shape)
            x = np.interp(freq_f[idx], fd[i:i + 3],
                          inter_value).astype(np.float32)
            if norm and len(idx):
                # x /= (fd[i + 2] - fd[i]) / sr * in_channels
                x /= x.sum()  # energy normalization

            if len(idx) == 0 and len(values) and len(values[-1]):
                # low resolution in the lower frequency (for spec)/ highter frequency (for ceps),
                # some filterbanks will not get any bin index, so we copy the indexes from the previous iteration
                c = cols[-1].copy()
                r = rows[-1].copy()
                r[:] = i
                x = values[-1].copy()

            cols.append(c)
            rows.append(r)
            values.append(x)

        cols, rows, values = np.concatenate(
            cols), np.concatenate(rows), np.concatenate(values)
        self.filter_f_weight[rows, cols] = torch.from_numpy(values)

        idxs = np.digitize(freq_t, fd)
        cols, rows, values = [], [], []
        for i in range(num_filters - 1, -1, -1):
            idx = np.where((idxs == i + 1) | (idxs == i + 2))[0]
            c = idx + 1
            r = np.broadcast_to(i, idx.shape)
            x = np.interp(freq_t[idx], fd[i:i + 3],
                          inter_value).astype(np.float32)
            if norm and len(idx):
                # x /= (1 / fd[i] - 1 / fd[i + 2]) * sr
                x /= x.sum()

            if len(idx) == 0 and len(values) and len(values[-1]):
                c = cols[-1].copy()
                r = rows[-1].copy()
                r[:] = i
                x = values[-1].copy()

            cols.append(c)
            rows.append(r)
            values.append(x)

        cols, rows, values = np.concatenate(
            cols), np.concatenate(rows), np.concatenate(values)
        self.filter_t_weight[rows, cols] = torch.from_numpy(values)

    def forward(self, ceps, spec):
        ceps, spec = ceps[:, :self.effected_dim,
                          :], spec[:, :self.effected_dim, :]

        spec = F.conv1d(spec, self.filter_f_weight.unsqueeze(2))
        ceps = F.conv1d(ceps, self.filter_t_weight.unsqueeze(2))
        return spec * ceps


if __name__ == "__main__":
    from torchinfo import summary
    import librosa
    from librosa import display
    import matplotlib.pyplot as plt
    mlc = MLC(44100, 8192, 1024, g=[
              0.1, 0.9, 0.9, 0.7, 0.8, 0.5], hipass_f=80, lowpass_t=1000 / 800)
    x = torch.randn(2, 44100 * 2)  # .cuda()
    summary(mlc, input_data=x, device='cpu')

    cfp = CFP(8192, 44100, norm=True, division=4, start_midi=40, end_midi=84)
    x = torch.rand(2, 8192, 165)
    summary(cfp, input_data=(x, x), device='cpu')

    plt.imshow(cfp.filter_f_weight.numpy()[
               :, :500], aspect='auto', origin='lower')
    plt.savefig("filter_f.jpeg")

    plt.imshow(cfp.filter_t_weight.numpy()[
               :, :], aspect='auto', origin='lower')
    plt.savefig("filter_t.jpeg")

    y, sr = librosa.load(librosa.ex('trumpet'), sr=44100)
    print(y.shape, sr)

    y = torch.Tensor(y[None, :])
    pitch = cfp(*mlc(y))
    print(pitch.shape, pitch.max())

    fig, ax = plt.subplots()
    img = display.specshow(
        pitch.numpy()[0], x_axis='time', ax=ax, sr=44100, hop_length=512)
    plt.savefig("cfp.jpeg")
    # im = Image.fromarray(np.round(cfp.filter_f_weight.numpy() * 255).astype('uint8'))
    # im.save("filter_f.jpeg")
