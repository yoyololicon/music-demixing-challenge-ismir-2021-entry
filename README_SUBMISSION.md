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

1. A X-UMX model [2] using the weights from the official baseline as initial weights, continue training with a modified version of **Combinational Multi-Domain Loss** from [2]. The modifications including running our own implementation of differentiable **Multi-Channel Wiener Filter** [3] just before computing the loss function, and calculating the frequency domain L2 loss using raw complex value.

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

Please describe here how your submission can be reproduced.

## How to reproduce the training

Please describe here how your model could be trained.

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