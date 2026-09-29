# EEG decoding baselines: stimulus type and sentiment

Small models that decode, from 4 seconds of EEG:
- **stimulus type**: was the person looking at an image, watching a video, or viewing an action image?
- **sentiment**: was the stimulus emotion positive or negative?

They were trained on a private single-subject dataset (30-channel Emotiv FLEX2, 256 Hz, 1,028
trials over 3 recording days). The dataset is not included.

| Model | Input | Parameters |
|---|---|---|
| **Logistic regression** | 600 band-power features (differential entropy, 4 × 1 s windows × 5 bands × 30 channels) | 1,803 / 601 |
| **EEGNet** ([Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c)) | raw EEG, 0–4 s after stimulus onset, band-pass 0.5–45 Hz, z-scored per channel | 3,123 / 2,610 |

Parameters are given as stimulus type / sentiment. These two model families ranked first and
second among everything we tried, including pretrained EEG foundation models (LaBraM, EEGPT).

## Checkpoints

| Folder | Task | Trained on | Score, random 80/20 split (LR / EEGNet) |
|---|---|---|---|
| `checkpoints/stimulus_full/` | stimulus type (3 classes, chance 0.33) | all 1,028 trials | **0.664 ± 0.027** / 0.610 ± 0.023 |
| `checkpoints/sentiment_full/` | positive vs negative (chance 0.50) | all 803 labelled trials | **0.593 ± 0.031** / 0.515 ± 0.040 |
| `checkpoints/stimtype/` | stimulus type, session-normalised variant | all 1,028 trials | 0.574 / — (built for new sessions, see below) |

Each folder holds `logreg.joblib`, `eegnet_seed{1,2,3}.pt` and `config.json` (the exact setup).
Scores are balanced accuracy (mean recall over classes, so guessing the largest class scores
chance), averaged over 5 random stratified 80/20 splits. The final checkpoints are then
retrained on the full data. Logistic regression is the better model on both tasks.

## Results under stricter tests

In a random split, test trials come from the same recording sessions as training trials.
The same models were also tested on data from sessions they never saw:

| Test | Stimulus type: LR / EEGNet | Sentiment: LR / EEGNet |
|---|---|---|
| Random 80/20 split | **0.664** / 0.610 | **0.593** / 0.515 |
| Calibration: first 70% of each session → last 30% | 0.483 / **0.522** | 0.550 / 0.522 |
| Held-out day: days 1–2 → day 3 | **0.506** / 0.468 | 0.495 / 0.563 |
| Leave one recording out | 0.399 / **0.508** | 0.521 / 0.536 |

- **Stimulus type** stays above chance (0.33) in every test, so it works on new sessions. For a
  new session, EEGNet (0.508) or the session-normalised model in `checkpoints/stimtype/` is
  the better choice.
- **Sentiment** falls to chance (0.50) on new sessions. Positive trials are much more common
  among images than among action images, so predicting sentiment from the stimulus type alone,
  with no EEG, already scores 0.628 on the random split. The sentiment checkpoints are best
  read as a baseline for this dataset.
- Training on all three days beats dropping the first day: on the same test recordings,
  stimulus-type EEGNet scores 0.576 with day 1 in training and 0.501 without it.

## Usage

```bash
pip install -r requirements.txt
python predict.py --ckpt checkpoints/stimulus_full  --input my_session.npy
python predict.py --ckpt checkpoints/sentiment_full --input my_session.npy
```

`my_session.npy` holds trials with shape `(n_trials, 30, 1408)`: 30 channels in the Emotiv
FLEX2 order, 256 Hz, from −0.5 s to +5.0 s around stimulus onset. The output is a CSV with the
predicted class and the class probabilities from both models (EEGNet averages its 3 seeds).
For `checkpoints/stimtype/`, pass one whole session (at least about 30 trials): its logistic
regression normalises features across the session.

## Files

| File | Purpose |
|---|---|
| `checkpoints/` | the three checkpoint folders above |
| `predict.py` | inference |
| `features.py`, `models.py` | preprocessing, EEGNet, checkpoint loading |
| `train.py` | training and every evaluation above (needs dataset access): `--configs R,ALL --lr_norm none` reproduces `*_full`; `A,B,C,D,E,CAL,ALL` the stricter tests and `stimtype/` |
| `split_comparison.py` | random vs calibration split comparison for logistic regression |
| `results/` | summaries and per-split results for every test |

EEGNet stops training on a 15% holdout of its training data and never sees the test data.
