# Submission

## Submission Summary

* Submission ID: 151898
* Submitter: yoyololicon
* Final rank: 4th place on leaderboard A
* Final scores on MDXDB21:

  | SDR_song | SDR_bass | SDR_drums | SDR_other | SDR_vocals |
  | :---:    | :---:    | :---:     | :---:     | :---:      |
  | 6.649    | 6.993    | 7.018     | 4.901     | 7.686      |

## Model Summary

Our final winning approach blends the outputs from three models, which are:

1. model 1: A X-UMX model [2] which is initialized with the weights of the official baseline, and is fine-tuned with a modified **Combinational Multi-Domain Loss** from [2]. In particular, we implement and apply a differentiable **Multichannel Wiener Filter** (MWF) [3] before the loss calculation, and compute the frequency-domain L2 loss with raw complex values.

2. model 2: A U-Net which is similar to **Spleeter** [4], where all convolution layers are replaced by D3 Blocks from [5], and two layers of 2D local attention are applied at the bottleneck similar to [6].

3. model 3: A modified version of **Demucs** [7], where the original decoding module is replaced by four independent decoders, each of which corresponds to one source.

We didn't encounter overfitting in our pilot experiments, so we used the full musdb training set for all the models above, and stopped training upon convergence of the loss function.

The weights of the three outputs are determined empirically:

|         | Drums | Bass | Other | Vocals |
|---------|-------|------|-------|--------|
| model 1 | 0.2   | 0.1  | 0     | 0.2    |
| model 2 | 0.2   | 0.17 | 0.5   | 0.4    |
| model 3 | 0.6   | 0.73 | 0.5   | 0.4    |

For the spectrogram-based models (model 1 and 2), we apply MWF to the outputs with one iteration before the fusion.

[1] Stöter, Fabian-Robert, et al. "Open-unmix-a reference implementation for
    music source separation." Journal of Open Source Software 4.41 (2019): 1667.

[2] Sawata, Ryosuke, et al. "All for One and One for All: Improving Music Separation by Bridging Networks." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.

[3] Antoine Liutkus, & Fabian-Robert Stöter. (2019). sigsep/norbert: First official Norbert release (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.3269749

[4] Hennequin, Romain, et al. "Spleeter: a fast and efficient music source separation tool with pre-trained models." Journal of Open Source Software 5.50 (2020): 2154.

[5] Takahashi, Naoya, and Yuki Mitsufuji. "D3net: Densely connected multidilated densenet for music source separation." arXiv preprint arXiv:2010.01733 (2020).

[6] Wu, Yu-Te, Berlin Chen, and Li Su. "Multi-Instrument Automatic Music Transcription With Self-Attention-Based Instance Segmentation." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2796-2809.

[7] Défossez, Alexandre, et al. "Music source separation in the waveform domain." arXiv preprint arXiv:1911.13254 (2019).

# Reproduction

## How to reproduce the submission

Our submission can be reproduced by:

1. Clone our submission [repo](https://gitlab.aicrowd.com/yoyololicon/music-demixing-challenge-starter-kit).  Make sure you have installed `git-lfs` before proceeding.


```commandline
git clone http://gitlab.aicrowd.com/yoyololicon/music-demixing-challenge-starter-kit.git
cd music-demixing-challenge-starter-kit/
```

2. Checkout to the winning submission tag.

```commandline
git checkout submission-fusion-3model-4
```

3. Install requirements.

```commandline
pip install -r requirements.txt
```

4. Use `python predict.py` to generate predictions on test data. The remaining steps are same as [official starter kit](https://github.com/AIcrowd/music-demixing-challenge-starter-kit).


## How to reproduce the training

### Install Requirements / Build Virtual Environment

We recommend using conda.

```commandline
conda env create -f environment.yml
conda activate demixing
```

### Prepare Data

Please download [musdb](https://zenodo.org/record/3338373), and edit the `"root"` parameters in all the json files listed under `configs/` to the path where you have the dataset.

### Training Model 1

First download the pre-trained model:

```commandline
wget https://zenodo.org/record/4740378/files/pretrained_xumx_musdb18HQ.pth
```

Copy the weights for initializing our model:

```commandline
python xumx_weights_convert.py pretrained_xumx_musdb18HQ.pth xumx_weights.pth
```

Start training!

```commandline
python train.py configs/x_umx_mwf.json --weights xumx_weights.pth
```

Checkpoints will be located under `saved/`.
The config was set to run on a single RTX 3070.

### Training Model 2


```commandline
python train.py configs/unet_attn.json --device_ids 0 1 2 3
```

Checkpoints will be located under `saved/`.
The config was set to run on four Tesla V100.

### Training Model 3


```commandline
python train.py configs/demucs_split.json
```

Checkpoints will be located under `saved/`.
The config was set to run on a single RTX 3070, using gradient accumulation and mixed precision training.

### Tensorboard Logging

You can monitor the training process using tensorboard:

```commandline
tesnorboard --logdir runs/
```

### Inference

After completing [How to reproduce the submission](#how-to-reproduce-the-submission), replace the jitted model listed under `your-cloned-submission-repo-dir/models/*` with the saved checkpoints.

```commandline
python jit_convert.py configs/x_umx_mwf.json saved/CrossNet Open-Unmix_checkpoint_XXX.pt your-cloned-submission-repo-dir/models/xumx_mwf_v4.pth

python jit_convert.py configs/unet_attn.json saved/UNet Attention_checkpoint_XXX.pt your-cloned-submission-repo-dir/models/unet_test.pth

python jit_convert.py configs/demucs_split.json saved/DemucsSplit_checkpoint_XXX.pt your-cloned-submission-repo-dir/models/demucs.pth
```

# License

MIT License

Copyright (c) [2021] [Chin-Yun Yu]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
