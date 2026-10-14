"""
Emotion weights v2 for the texture pipeline (2026-10-12): Euclidean alignment + ERP shape + shrinkage LDA.

Best of everything tried on clean data (370 clean image trials, scored on recordings never seen in training):
  single trial          26.3% top-emotion accuracy (chance 20%, p = 0.0006)
  block of 2 trials     28.6% (p = 0.01)
  block of 8 trials     33.4% (p = 0.01)
  on pictures never seen in training it is not yet significant (22.6%, p = 0.12), so part of the
  signal may be specific pictures rather than the emotion.
The probability VALUES are over-confident; mode="calibrated" uses the temperature fitted on held-out
predictions. sharp only adds visual contrast.

Use (a session = trials recorded with one headset placement):
    ew = EmotionWeightsV2()
    ew.calibrate(session_epochs)              # (n, 30, 1408) epochs of that session, labels not needed
    w = ew.weights(epoch)                     # one trial -> {"Joy": .., ...}
    w = ew.weights_block([e1, e2, e3, e4])    # average of a block of trials (more accurate)
"""
import os, numpy as np, joblib
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
EMOS = ["Joy", "Tenderness", "Serenity", "Nostalgia", "Melancholy"]
_SOS_COV = butter(4, [1, 45], btype="band", fs=256, output="sos")
_SOS_ERP = butter(4, [1, 30], btype="band", fs=256, output="sos")


def session_whitener(epochs):
    """R^(-1/2), R = mean 1-45 Hz spatial covariance (0-4 s) over the session's trials."""
    Cs = []
    for e in epochs:
        x = sosfiltfilt(_SOS_COV, np.asarray(e, np.float64), axis=1)[:, 128:1152]; x = x - x.mean(1, keepdims=True)
        Cs.append(x @ x.T / x.shape[1])
    w, V = np.linalg.eigh(np.mean(Cs, 0))
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def erp_features(e):
    x = sosfiltfilt(_SOS_ERP, np.asarray(e, np.float64), axis=1); x = x - x[:, :128].mean(1, keepdims=True)
    return x[:, 128:368].reshape(30, 20, 12).mean(2).reshape(-1)


class EmotionWeightsV2:
    def __init__(self, path=os.path.join(HERE, "checkpoints", "emotion5", "emotion_weights_v2.joblib"), mode="model", sharp=1.0):
        b = joblib.load(path); self.lda, self.scaler, self.T = b["lda"], b["scaler"], b["temperature"]
        self.mode, self.sharp, self.W = mode, sharp, None

    def calibrate(self, session_epochs):
        self.W = session_whitener(session_epochs); return self

    def _logp(self, epochs):
        if self.W is None:
            raise RuntimeError("call calibrate(session_epochs) first: the model needs the session's alignment")
        F = np.stack([erp_features(self.W @ np.asarray(e, np.float64)) for e in epochs])
        return np.log(np.clip(self.lda.predict_proba(self.scaler.transform(F)), 1e-9, 1))

    def _to_weights(self, logp, mode, sharp):
        z = logp / (self.T if (mode or self.mode) == "calibrated" else 1.0) * (sharp if sharp is not None else self.sharp)
        z = z - z.max(); w = np.exp(z); w /= w.sum()
        return dict(zip(EMOS, np.round(w, 4)))

    def weights(self, epoch, mode=None, sharp=None):
        return self._to_weights(self._logp([epoch])[0], mode, sharp)

    def weights_block(self, epochs, mode=None, sharp=None):
        p = np.exp(self._logp(epochs)).mean(0)                 # average probabilities over the block
        return self._to_weights(np.log(p), mode, sharp)
