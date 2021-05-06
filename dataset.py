import musdb
from torch.utils.data import Dataset
import random
import numpy as np
import soundfile as sf
import os
import yaml
from tqdm import tqdm


__all__ = ['FastMUSDB']


def _get_nframes(info_str: str):
    try:
        return int(info_str.split('frames  : ')[1].split('\n')[0])
    except:
        byte_sec = int(info_str.split(
            'Bytes/sec     : ')[1].split('\n')[0])
        data = int(info_str.split('data : ')[1].split('\n')[0])
        sr = int(info_str.split('Sample Rate   : ')[1].split('\n')[0])
        return int(data / byte_sec * sr)


class FastMUSDB(Dataset):
    def __init__(self,
                 root=None,
                 subsets=['train', 'test'],
                 split=None,
                 seq_duration=6.0,
                 samples_per_track=64,
                 random=False,
                 random_track_mix=False,
                 ):
        self.root = os.path.expanduser(root)
        self.seq_duration = seq_duration
        self.subsets = subsets
        self.sr = 44100
        self.segment = int(self.seq_duration * self.sr)
        self.split = split
        self.samples_per_track = samples_per_track
        self.random_track_mix = random_track_mix
        self.random = random
        self.sources = ['drums', 'bass', 'other', 'vocals']

        setup_path = os.path.join(
            musdb.__path__[0], 'configs', 'mus.yaml'
        )
        with open(setup_path, 'r') as f:
            self.setup = yaml.safe_load(f)

        self.tracks, self.track_lenghts = self.load_mus_tracks(
            self.sr, self.subsets, self.split)

        if self.seq_duration <= 0:
            self._size = len(self.tracks)
        elif self.random:
            self._size = len(self.tracks) * self.samples_per_track
        else:
            chunks = [l // self.segment for l in self.track_lenghts]
            cum_chunks = np.cumsum(chunks)
            self.cum_chunks = cum_chunks
            self._size = cum_chunks[-1]

    def load_mus_tracks(self, sr, subsets=None, split=None):
        if subsets is not None:
            if isinstance(subsets, str):
                subsets = [subsets]
        else:
            subsets = ['train', 'test']

        if subsets != ['train'] and split is not None:
            raise RuntimeError(
                "Subset has to set to `train` when split is used")

        print("Gathering files ...")
        tracks = []
        track_lengths = []
        for subset in subsets:
            subset_folder = os.path.join(self.root, subset)
            for _, folders, files in tqdm(os.walk(subset_folder)):
                # parse pcm tracks and sort by name
                for track_name in sorted(folders):
                    if subset == 'train':
                        if split == 'train' and track_name in self.setup['validation_tracks']:
                            continue
                        elif split == 'valid' and track_name not in self.setup['validation_tracks']:
                            continue

                    track_folder = os.path.join(subset_folder, track_name)
                    # add track to list of tracks
                    tracks.append(track_folder)

                    f_obj = sf.SoundFile(os.path.join(
                        track_folder, 'mixture.wav'))
                    assert f_obj.samplerate == sr

                    track_lengths.append(_get_nframes(f_obj.extra_info))
                    f_obj.close()

        return tracks, track_lengths

    def __len__(self):
        return self._size

    def _get_random_track_idx(self):
        return random.randrange(len(self.tracks))

    def _get_random_start(self, length):
        return random.randrange(length - self.segment + 1)

    def _get_track_from_chunk(self, index):
        track_idx = np.digitize(index, self.cum_chunks)
        if track_idx > 0:
            chunk_start = (index - self.cum_chunks[track_idx]) * self.segment
        else:
            chunk_start = index * self.segment
        return self.tracks[track_idx], chunk_start

    def __getitem__(self, index):
        stems = []
        if self.seq_duration <= 0:
            folder_name = self.tracks[index]
            x = sf.read(
                os.path.join(folder_name, 'mixture.wav'),
                dtype='float32', always_2d=True
            )[0].T
            for s in self.sources:
                source_name = os.path.join(folder_name, s + '.wav')
                audio = sf.read(
                    source_name,
                    dtype='float32', always_2d=True
                )[0].T
                stems.append(audio)
        else:
            if self.random:
                track_idx = index // self.samples_per_track
                folder_name, chunk_start = self.tracks[track_idx], self._get_random_start(
                    self.track_lenghts[track_idx])
            else:
                folder_name, chunk_start = self._get_track_from_chunk(index)
            for s in self.sources:
                g = 1
                swap = False
                if self.random_track_mix and self.random:
                    track_idx = self._get_random_track_idx()
                    folder_name, chunk_start = self.tracks[track_idx], self._get_random_start(
                        self.track_lenghts[track_idx])
                    g = random.uniform(0.25, 1.25)
                    if random.random() > 0.5:
                        swap = True
                source_name = os.path.join(folder_name, s + '.wav')
                audio = sf.read(
                    source_name, frames=self.segment, start=chunk_start,
                    dtype='float32', always_2d=True, fill_value=0.
                )[0].T
                audio *= g
                if swap:
                    audio = np.flip(audio, 0)
                stems.append(audio)
            if self.random_track_mix and self.random:
                x = sum(stems)
            else:
                x = sf.read(
                    os.path.join(folder_name, 'mixture.wav'),
                    frames=self.segment, start=chunk_start,
                    dtype='float32', always_2d=True, fill_value=0.
                )[0].T

        y = np.stack(stems)
        return x.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    #dataset = MUSDataset(os.path.expanduser("~/Datasets/musdb18hq"))
    dataset = FastMUSDB(os.path.expanduser(
        "~/Datasets/musdb18hq"), seq_duration=5, random=False, random_track_mix=False)
    loader = DataLoader(dataset, 4, True)

    print(len(loader), dataset._size)
    for i, (x, y) in enumerate(loader):
        print(i, x.shape, y.shape)
