import torch
import torchaudio
    

class SpeedPerturb(torch.nn.Module):
    def __init__(
        self, orig_freq=44100, speeds=[90, 100, 110]
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.resamplers = []
        self.speeds = speeds
        for s in self.speeds:
            new_freq = self.orig_freq * s // 100
            self.resamplers.append(
                    torchaudio.transforms.Resample(self.orig_freq, new_freq))

    def forward(self, audios):
        """
        Args:
            audios (torch.Tensor): (B, Num_sources, Num_channels, L)
        Return:
            perturbed_audios (torch.Tensor): (B, Num_sources, Num_channels, L')
        """
        # Perform source-wise random perturbation
        new_audios = []
        for i in range(audios.shape[1]):
            samp_index = torch.randint(len(self.speeds), (1,))[0]
            perturbed_audio = self.resamplers[samp_index](audios[:, i].contiguous())
            new_audios.append(perturbed_audio)
            if i == 0:
                min_len = perturbed_audio.shape[-1]
            else:
                if perturbed_audio.shape[-1] < min_len:
                    min_len = perturbed_audio.shape[-1]

        perturbed_audios = torch.zeros(
            audios.shape[0],
            audios.shape[1],
            2,
            min_len,
            device=audios.device,
            dtype=torch.float,
            )

        for i, _ in enumerate(new_audios):
            perturbed_audios[:, i] = new_audios[i][:, :, 0:min_len]

        return perturbed_audios

class Transforms:
    def __init__(self, orig_freq=44100):
        self.speed_perturb = SpeedPerturb(orig_freq)
        self.transforms = [self.speed_perturb]

    def __call__(self, audio):
        for trans in self.transforms:
            audio = trans(audio)
        return audio
