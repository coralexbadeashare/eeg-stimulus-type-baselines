# EEG stimulus-type decoding: baseline checkpoints

Two small models that tell, from 4 seconds of EEG, whether a person was looking at an
**image**, watching a **video**, or viewing an **action image**. Trained on a private
single-subject dataset (30-channel Emotiv FLEX2, 256 Hz, 1,028 trials over 3 recording days).
The dataset is not included.

| Model | Input | Parameters |
|---|---|---|
| **EEGNet** ([Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c)) | raw EEG, 0–4 s after stimulus onset, band-pass 0.5–45 Hz, z-scored per channel | 3,123 |
| **Logistic regression** | 600 band-power features (differential entropy, 4 × 1 s windows × 5 bands × 30 channels), z-scored within each recording session | 1,803 |

Both checkpoints are trained on **all three days**. These two ranked first and second among
everything we tried, including pretrained EEG foundation models (LaBraM, EEGPT).

## Results

Balanced accuracy over 3 classes. Chance is 0.33.

| How the model is tested | Question it answers | Logistic regression | EEGNet (3 seeds) |
|---|---|---|---|
| Random trial split (80/20) | new trials, from recordings already seen in training | **0.661 ± 0.025** | — |
| Calibration: first 70% of each recording → last 30% | calibrate at the start of a session, then predict | **0.483** (label-shuffle p = 0.01) | — |
| Held-out day: train days 1–2 → test day 3 | a new day | **0.506** (label-shuffle p = 0.02) | 0.468 ± 0.080 |
| Leave one recording out, all days | a new recording (a fresh headset placement) | 0.399 | **0.508 ± 0.013** |

**How to read this.** The same model scores lower the harder the test. The random split is the
most optimistic: trials from one recording share a day, a headset placement and a protocol, so
part of that score is recognising the recording. Knowing only which recording a trial came from
already scores 0.45 on that split. The leave-one-recording-out row is the honest estimate for a
new session. There, EEGNet is the best model and is stable across seeds.

Training on all three days is better than dropping the first day. On the same day-2/3 test
recordings, EEGNet scores 0.576 with day 1 in training and 0.501 without it.

### What did not work

On a new day or recording, **positive vs negative emotion** is not decodable from this data:
0.50–0.56 against a chance level of 0.50. Positive trials are much more common among images
than among action images, so predicting emotion from the stimulus type alone, with no EEG,
scores 0.65. No emotion checkpoint is included.

## Notes on method

- **Normalise per session.** Absolute band power shifts between recording days. Without
  per-session z-scoring, the logistic regression drops to 0.19 on a held-out day.
- **Pick epochs on training data only.** EEGNet stops on a 15% holdout of its training set,
  never on the test data.
- **Controls.** Every score is checked against label-shuffled models. For emotion, it is also
  checked against a stimulus-type-only baseline.

## Usage

```bash
pip install -r requirements.txt
python predict.py --ckpt checkpoints/stimtype --input my_session.npy
```

`my_session.npy` holds trials from one session, with shape `(n_trials, 30, 1408)`: 30 channels
in the Emotiv FLEX2 order, 256 Hz, from −0.5 s to +5.0 s around stimulus onset. The logistic
regression normalises across the trials you pass, so give it a whole session (at least about
30 trials). EEGNet works per trial and averages its 3 seeds. The output is a CSV with the
predicted class and the class probabilities from both models.

## Files

| File | Purpose |
|---|---|
| `checkpoints/stimtype/` | `eegnet_seed{1,2,3}.pt`, `logreg.joblib`, `config.json` (exact training setup) |
| `predict.py` | inference on a session |
| `features.py`, `models.py` | preprocessing, EEGNet, checkpoint loading |
| `train.py` | training and every evaluation setup above (needs dataset access) |
| `split_comparison.py` | random vs calibration split comparison (needs dataset access) |
| `results/` | per-setup summaries and per-fold results |
