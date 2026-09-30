"""
Preprocessing shared by training and inference.

Input everywhere is one trial (`eeg_epoch`), shape (30, 1408),
microvolts, 256 Hz, from -0.5 s to +5.0 s around stimulus onset. Both models use the 4 s
after onset (samples 128:1152).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

SFREQ = 256
ONSET, WIN = 128, 1024                       # 0-4 s after stimulus onset
N_CH = 30
BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 100.0)]

# valence labels (the 803-trial set); 03_NOSTALGIA and unlabeled trials are excluded
POSITIVE = {"01_JOY", "02_TENDERNESS", "04_SERENITY", "happy", "excited",
            "joy", "serenity", "tenderness", "exaltation", "relax"}
NEGATIVE = {"05_MELANCHOLY", "anxiety", "tired", "frustration", "bored",
            "angry", "anguish"}
CLASSES = ["negative", "positive"]


def check_epoch(eeg):
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[0] != N_CH or eeg.shape[1] < ONSET + WIN:
        raise ValueError(f"expected (30, >= {ONSET + WIN}) epoch at 256 Hz starting 0.5 s before onset, got {eeg.shape}")
    return eeg


def is_empty(epoch, tol=1e-3):
    """True when every channel is flat after onset (all-zero trial: the headset was not recording)."""
    return float(np.asarray(epoch, dtype=np.float32)[:, ONSET:ONSET + WIN].std(1).max()) < tol


def _de(x):
    return 0.5 * np.log(2.0 * np.pi * np.e * (np.var(x, axis=-1, ddof=1) + 1e-10))


def extract_de(eeg, bands=BANDS):
    """(30, 1024) post-onset window -> (4 windows, 5 bands, 30 ch) differential entropy."""
    out = np.empty((4, len(bands), eeg.shape[0]), dtype=np.float32)
    for b, (_, lo, hi) in enumerate(bands):
        filt = sosfiltfilt(butter(5, [lo, hi], btype="band", fs=SFREQ, output="sos"), eeg, axis=-1)
        for w in range(4):
            out[w, b] = _de(filt[:, w * SFREQ:(w + 1) * SFREQ])
    return out


def de_features(epoch):
    """Features for the logistic-regression model: 600 values per trial."""
    return extract_de(check_epoch(epoch)[:, ONSET:ONSET + WIN]).reshape(-1)


def raw_input(epoch):
    """Input for EEGNet: band-pass 0.5-45 Hz, 4 s post-onset, z-score per channel."""
    sos = butter(4, [0.5, 45], btype="band", fs=SFREQ, output="sos")
    x = sosfiltfilt(sos, check_epoch(epoch), axis=1)[:, ONSET:ONSET + WIN]
    return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)).astype(np.float32)
