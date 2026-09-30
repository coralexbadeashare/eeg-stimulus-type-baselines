# EEG decoding baselines: stimulus type and sentiment

Small models that decode, from 4 seconds of EEG:
- **stimulus type**: was the person looking at an image, watching a video, or viewing an action image?
- **sentiment**: was the stimulus emotion positive or negative?

They were trained on a private single-subject dataset (30-channel Emotiv FLEX2, 256 Hz, 3
recording days). The dataset is not included.

## Data preparation

- **ZUNA-cleaned.** Every recording was denoised with the ZUNA EEG diffusion denoiser before
  the trials were cut.
- **Trials re-aligned.** Each trial was assigned to the recording whose time window actually
  contains it.
- **No-signal trials dropped.** Trials with no covering recording, or flat on every channel,
  were removed, leaving **910 trials**: 377 image, 185 video, 348 action image. For sentiment,
  712 labelled trials: 488 positive, 224 negative.
- **Quality checked before training.** No flat trials, no NaN, 99th-percentile amplitude
  430 µV, no time shift against the uncleaned signal.

Garbage in, garbage out: on the uncleaned version, 22% of trials were all zeros. That lowered
stimulus-type scores, and it inflated sentiment scores because most empty trials were
(mostly positive) images.

## Models

| Model | Input | Parameters (stimulus / sentiment) |
|---|---|---|
| **Logistic regression** | 600 band-power features: differential entropy, 4 × 1 s windows × 5 bands (delta, theta, alpha, beta, gamma) × 30 channels | 1,803 / 601 |
| **EEGNet** ([Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c)) | raw EEG, 0–4 s after stimulus onset, band-pass 0.5–45 Hz, z-scored per channel | 3,123 / 2,610 |

These two model families ranked first and second among everything we tried, including
pretrained EEG foundation models (LaBraM, EEGPT).

## Checkpoints

| Folder | Task | Trained on | Random 80/20 split: LR / EEGNet |
|---|---|---|---|
| `checkpoints/stimulus_full/` | stimulus type (3 classes, chance 0.33) | all 910 trials | **0.700 ± 0.027** / 0.637 ± 0.017 |
| `checkpoints/sentiment_full/` | positive vs negative (chance 0.50) | all 712 labelled trials | **0.561 ± 0.026** / 0.526 ± 0.029 |

Each folder holds `logreg.joblib`, `eegnet_seed{1,2,3}.pt` and `config.json` (the exact setup).
Scores are balanced accuracy (mean recall over classes, so guessing the largest class scores
chance), averaged over 5 random stratified 80/20 splits. The checkpoints are then retrained on
the full data.

## Results under stricter tests

In a random split, test trials come from the same recording sessions as training trials. The
same models were also tested on sessions they never saw:

| Test | Stimulus type: LR / EEGNet | Sentiment: LR / EEGNet |
|---|---|---|
| Random 80/20 split | **0.700** / 0.637 | **0.561** / 0.526 |
| Calibration: first 70% of each session → last 30% | **0.584** / 0.503 | 0.538 / 0.498 |
| Held-out day: days 1–2 → day 3 | 0.563 / **0.576** | 0.533 / 0.514 |
| Leave one recording out | 0.563 / **0.576 ± 0.005** | 0.544 / 0.510 |

- **Stimulus type** is well above chance (0.33) in every test, including sessions never seen
  in training. For a new session, EEGNet is the most reliable (0.576, stable across seeds).
- **Sentiment** is weak: 0.51–0.56 against a chance level of 0.50. Positive trials are much
  more common among images than among action images, so predicting sentiment from the
  stimulus type alone, with no EEG, scores 0.652. Measured within a single stimulus type, the
  models score 0.52–0.59. The sentiment checkpoints are a baseline, not a working emotion
  decoder.
- **Training on all three days is better** than dropping day 1: on the same day-2/3 test
  recordings, stimulus-type EEGNet scores 0.614 with day 1 and 0.537 without it.

## Usage

```bash
pip install -r requirements.txt
python predict.py --ckpt checkpoints/stimulus_full  --input my_session.npy
python predict.py --ckpt checkpoints/sentiment_full --input my_session.npy
```

`my_session.npy` holds trials with shape `(n_trials, 30, 1408)`: 30 channels in the Emotiv
FLEX2 order, 256 Hz, µV, from −0.5 s to +5.0 s around stimulus onset. Clean the input with
ZUNA first, the same way as the training data. Flat (no-signal) trials are detected and get no
prediction. The output is a CSV with the predicted class and class probabilities from both
models (EEGNet averages its 3 seeds).

## Files

| File | Purpose |
|---|---|
| `checkpoints/` | the two checkpoint folders above |
| `predict.py` | inference |
| `features.py`, `models.py` | preprocessing, flat-trial detection, EEGNet, checkpoint loading |
| `train.py` | training and every evaluation above (needs dataset access). `--configs R,ALL --lr_norm none` reproduces the checkpoints; the default configs run the stricter tests |
| `split_comparison.py` | random vs calibration split comparison for logistic regression |
| `results/` | summaries and per-split results (recording names anonymised) |

EEGNet stops training on a 15% holdout of its training data and never sees the test data.
