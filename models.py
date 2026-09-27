"""EEGNet (Lawhern et al., 2018) and checkpoint loaders for both models."""
import joblib
import numpy as np
import torch
import torch.nn as nn

from features import WIN, N_CH


class EEGNet(nn.Module):
    def __init__(self, n_ch=N_CH, n_t=WIN, n_cls=2, F1=8, D=2, F2=16, k=64, drop=0.5):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1))
        self.b2 = nn.Sequential(nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                                nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.b3 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                                nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        self.head = nn.Linear(F2 * (n_t // 32), n_cls)

    def forward(self, x):                    # x: (batch, 30, 1024)
        return self.head(self.b3(self.b2(self.b1(x.unsqueeze(1)))).flatten(1))


def load_eegnet(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["eegnet"]
    m = EEGNet(n_cls=len(ck["classes"]), F1=cfg["F1"], D=cfg["D"], F2=cfg["F2"], k=cfg["k"], drop=cfg["drop"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def load_logreg(path):
    return joblib.load(path)


@torch.no_grad()
def eegnet_proba(model, X):
    """X: (n, 30, 1024) from features.raw_input -> (n, 2) probabilities [negative, positive]."""
    return torch.softmax(model(torch.from_numpy(np.asarray(X, dtype=np.float32))), 1).numpy()


def normalize_session(F):
    """z-score each DE feature across the trials of ONE recording session (no labels needed).
    The logistic-regression checkpoints were trained on features normalised this way, so give
    them a batch of trials from the same session (ideally the whole session, >= ~30 trials)."""
    F = np.asarray(F, dtype=np.float64)
    return (F - F.mean(0)) / (F.std(0) + 1e-6)


def logreg_proba(bundle, F_session):
    """F_session: (n, 600) DE features of one session, already passed through normalize_session."""
    return bundle["model"].predict_proba(bundle["scaler"].transform(np.asarray(F_session)))
