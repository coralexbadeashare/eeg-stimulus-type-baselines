"""Why a random trial split scores higher than a new recording.
Calibration split: per recording, first 70% of trials (by time) -> train, last 30% -> test.
Mimics real BCI use (calibrate at the start of a session, then predict). Compared with the
random split and with leave-one-recording-out, same logistic regression and DE features."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score as bal
from datasets import load_from_disk
from features import de_features, POSITIVE, NEGATIVE

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
ds = load_from_disk(a.data)
meta = ds.select_columns(["stimulus_type", "eeg_recording_id", "stimulus_onset_unix", "emotion"]).to_pandas()
with Pool(16) as p:
    DE_all = np.stack(p.map(de_features, [np.asarray(e, dtype=np.float32) for e in ds["eeg_epoch"]]))

def znorm(F, rec):
    F = F.copy()
    for r in np.unique(rec):
        m = rec == r; F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-6)
    return F

def fit_pred(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    return LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))

def calib_split(rec, onset, frac=0.7):
    tr, te = [], []
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]; idx = idx[np.argsort(onset[idx])]; k = int(round(frac * len(idx)))
        tr += list(idx[:k]); te += list(idx[k:])
    return np.array(tr), np.array(te)

def random_split(y, seed):
    rng = np.random.default_rng(seed); tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx); n = int(0.2 * len(idx)); te += list(idx[:n]); tr += list(idx[n:])
    return np.array(tr), np.array(te)

def rec_only(y, rec, tr, te):
    maj = pd.Series(y[tr]).groupby(rec[tr]).agg(lambda s: s.value_counts().idxmax())
    return np.array([maj.get(r, np.bincount(y[tr]).argmax()) for r in rec[te]])

def type_only(y, typ, tr, te):
    rate = pd.Series(y[tr]).groupby(typ[tr]).mean()
    return np.array([int(rate.get(t, y[tr].mean()) >= y[tr].mean()) for t in typ[te]])

def perm_p(X, y, tr, te, real, n=200):
    rng = np.random.default_rng(0); null = []
    for _ in range(n):
        yp = y.copy(); yp[tr] = rng.permutation(y[tr]); null.append(bal(y[te], fit_pred(X, yp, tr, te)))
    null = np.array(null); return null.mean(), (1 + (null >= real).sum()) / (1 + n)

out = []
for task in ["stimulus type (3 classes, chance 0.33)", "positive vs negative (chance 0.50)"]:
    if task.startswith("stimulus"):
        sel = np.arange(len(meta)); y = meta.stimulus_type.map({"image": 0, "video": 1, "action_image": 2}).values
    else:
        sel = np.where(meta.emotion.isin(POSITIVE | NEGATIVE))[0]; y = meta.emotion.iloc[sel].isin(POSITIVE).astype(int).values
    rec = meta.eeg_recording_id.values[sel]; onset = meta.stimulus_onset_unix.values[sel]; typ = meta.stimulus_type.values[sel]
    DE = DE_all[sel]; DEr = znorm(DE, rec)
    tr, te = calib_split(rec, onset)
    row = dict(task=task, n_train=len(tr), n_test=len(te))
    for name, X in [("no_norm", DE), ("per_rec_norm", DEr)]:
        pr = fit_pred(X, y, tr, te); s = bal(y[te], pr)
        nm, p = perm_p(X, y, tr, te, s)
        row[f"calibration_{name}"] = round(s, 3); row[f"calibration_{name}_perm"] = f"null {nm:.3f}, p={p:.3f}"
        rs = [bal(y[t2], fit_pred(X, y, t1, t2)) for t1, t2 in (random_split(y, s_) for s_ in range(1, 6))]
        row[f"random_{name}"] = round(float(np.mean(rs)), 3)
    row["calibration_recording_name_only"] = round(bal(y[te], rec_only(y, rec, tr, te)), 3)
    if task.startswith("positive"):
        row["calibration_stimulus_type_only"] = round(bal(y[te], type_only(y, typ, tr, te)), 3)
        pr = fit_pred(DEr, y, tr, te)
        row["calibration_within_type"] = {t: round(bal(y[te][typ[te] == t], pr[typ[te] == t]), 3) for t in ["image", "action_image", "video"]}
    out.append(row)
for r in out:
    print("\n==", r.pop("task"))
    for k, v in r.items(): print(f"  {k:<36} {v}")
