> Dear participants, for the final due diligence on your submission, please use the below README format to describe your submission. We also encourage you to add useful comments in your code to help streamline the review process.


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

Our final winning approach is blending the outputs from three types of model, which are:

1. A X-UMX model [2] using the weights from the official baseline as initial weights, continue training with a modified version of **Combinational Multi-Domain Loss** from [2]. The modifications including running our own implementation of differentiable **MultiChannel Wiener Filter** [3] just before computing the loss function, and calculating the frequency domain L2 loss using raw complex value.

2. A U-Net which structure is similar to **Spleeter** [4], with all convolution layers being replaced by D3 Blocks from [5], and apply 2 layers of 2D local attention at the bottle neck similar to [6].

3. A modified version of **Demucs** [7], with the docoder part being replaced by 4 independent decoders, each corresponding to one source.

In our early experiments we didn't observe any obvious overfitting, so we use the full musdb training set for all the models above, and stop training when the loss curve converge.

The blending weights were set empirically:

|         | Drums | Bass | Other | Vocals |
|---------|-------|------|-------|--------|
| model 1 | 0.2   | 0.1  | 0     | 0.2    |
| model 2 | 0.2   | 0.17 | 0.5   | 0.4    |
| model 3 | 0.6   | 0.73 | 0.5   | 0.4    |

For spectrogram-based models (model 1 and 2), we use 1 iteration of multichannel wiener filtering before blending.

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

1. Clone our submission [repo](https://gitlab.aicrowd.com/yoyololicon/music-demixing-challenge-starter-kit).


```commandline
git clone http://gitlab.aicrowd.com/yoyololicon/music-demixing-challenge-starter-kit.git
cd music-demixing-challenge-starter-kit/
git pull && git lfs pull
```

2. Checkout to the winning submission tag.

```commandline

git checkout submission-fusion-3model-4
```

3. Install requirements.

```commandline
pip install requirements.txt
```

4. Use `python predict.py` to generate prediction on test data. Other steps can be infer from [official starter kit](https://github.com/AIcrowd/music-demixing-challenge-starter-kit).


## How to reproduce the training

### Install Requirements / Build Virtual Environment

We recommend using conda.

```commandline
conda env create -f environment.yml
conda activate demixing
```

### Prepare Data

Please download [musdb](https://zenodo.org/record/3338373), and edit the `"root"` parameter in all the json files list under `configs/` to where you put the dataset .

### Training Model 1

First download the pre-trained model:

```commandline
wget https://zenodo.org/record/4740378/files/pretrained_xumx_musdb18HQ.pth
```

Copy the weights to match our model:

```commandline
python xumx_weights_convert.py pretrained_xumx_musdb18HQ.pth xumx_weights.pth
```

Start training!

```commandline
python train.py configs/x_umx_mwf.json --weights xumx_weights.pth
```

Checkpoints will be located at `saved/`.
The config was set to run on a single RTX 3070.

### Training Model 2


```commandline
python train.py configs/unet_attn.json --device_ids 0 1 2 3
```

Checkpoints will be located at `saved/`.
The config was set to run on 4 Tesla V100.

### Training Model 3


```commandline
python train.py configs/demucs_split.json
```

Checkpoints will be located at `saved/`.
The config was set to run on a single RTX 3070, using gradient accumulation and mixed precision training.

### Tensorboard Logging

You can monitor the training process using tensorboard:

```commandline
tesnorboard --logdir runs/
```

### Inference

After complete [How to reproduce the submission](#how-to-reproduce-the-submission), replace the jitted model listed under `your-cloned-submission-repo-dir/models/*` with the newly trained checkpoints.

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