"""
Train and evaluate the two baseline models (DE + logistic regression, EEGNet) on a private
single-subject EEG dataset, with test data always taken from recordings the model never saw.

Tasks
  valence   positive vs negative emotion (803 trials; see features.POSITIVE / NEGATIVE)
  stimtype  image vs video vs action image (all 1028 trials) -- a task with a known signal

Configurations (recording days: May-06, May-07, May-08)
  A  train May-06 + May-07          -> test May-08
  B  train May-07 only              -> test May-08
  C  May-08 only, leave-one-recording-out
  D  all days, leave-one-recording-out            (reference; also scored on May-07/08 folds)
  E  May-07 + May-08 only, leave-one-recording-out  (= D without the first day)
  CAL  calibration: first 70% of each recording (by time) -> train, last 30% -> test
       (how a BCI is used: calibrate at the start of a session, then predict)
  ALL  train on every trial -> the final checkpoints; expected accuracy on a new recording = D
  D vs E on the same May-07/08 test folds isolates what the first day adds or costs.

Evaluation rules
  - EEGNet picks its epoch on 15% of the TRAINING data (stratified), never on test.
  - Logistic regression: fixed C=0.5, class-balanced, no tuning. Its 600 DE features are
    z-scored within each recording first (labels unused): May-06/07 were exported already
    band-passed by EmotivPRO and May-08 raw, which shifts absolute band power between days.
    The un-normalised variant is reported too (logreg_no_session_norm).
  - Controls: label permutations for logistic regression; for valence also "stimulus type
    only, no EEG", because valence labels are unevenly spread across stimulus types.

Usage
  python train.py --task stimtype --data /path/to/dataset
  python train.py --task valence  --data /path/to/dataset
CPU is enough (EEGNet has 2.6k parameters).
"""
import os, json, time, argparse, numpy as np, pandas as pd
from multiprocessing import get_context
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import joblib

from features import de_features, raw_input, POSITIVE, NEGATIVE, ONSET, WIN, SFREQ

TASKS = {"valence": ["negative", "positive"], "stimtype": ["image", "video", "action_image"]}
EEGNET = dict(F1=8, D=2, F2=16, k=64, drop=0.5, lr=1e-3, wd=1e-2, batch=32,
              max_epochs=150, patience=25, val_frac=0.15, noise=0.1)
LOGREG = dict(C=0.5, class_weight="balanced", max_iter=5000)
LAST, MID = "May-08", "May-07"
W = {}                                        # worker globals


# ------------------------------------------------------------------ EEGNet worker
def _init(cache, n_cls):
    import torch
    torch.set_num_threads(6)
    W["X"] = np.load(cache + "_X.npy"); W["y"] = np.load(cache + "_y.npy"); W["n_cls"] = n_cls


def strat_holdout(y, frac, seed):
    rng = np.random.default_rng(seed); tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx); n = max(1, int(round(frac * len(idx))))
        va += list(idx[:n]); tr += list(idx[n:])
    return np.array(tr), np.array(va)


def train_eegnet(job):
    import torch, torch.nn as nn
    from models import EEGNet
    tag, tr, te, seed, ckpt, task = job
    X, y, n_cls = W["X"], W["y"], W["n_cls"]
    torch.manual_seed(seed)
    a, b = strat_holdout(y[tr], EEGNET["val_frac"], seed); itr, iva = tr[a], tr[b]
    m = EEGNet(n_cls=n_cls, **{k: EEGNET[k] for k in ("F1", "D", "F2", "k", "drop")})
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    cnt = np.bincount(y[itr], minlength=n_cls)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(len(itr) / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32))
    opt = torch.optim.AdamW(m.parameters(), lr=EEGNET["lr"], weight_decay=EEGNET["wd"])
    Xt = lambda i: torch.from_numpy(X[np.sort(i)])

    def run(idx):
        idx = np.sort(idx); m.eval()
        with torch.no_grad():
            out = torch.cat([m(Xt(idx[j:j + 256])) for j in range(0, len(idx), 256)])
        return crit(out, torch.from_numpy(y[idx])).item(), out.argmax(1).numpy(), idx

    log, best, best_ep, state, bad = [], -1.0, 0, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(1, EEGNET["max_epochs"] + 1):
        m.train(); order = rng.permutation(itr); tl, tt, tp = 0.0, [], []
        for j in range(0, len(order), EEGNET["batch"]):
            bi = np.sort(order[j:j + EEGNET["batch"]])
            xb = Xt(bi) + EEGNET["noise"] * torch.randn(len(bi), X.shape[1], X.shape[2])
            out = m(xb); loss = crit(out, torch.from_numpy(y[bi]))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(bi); tt += list(y[bi]); tp += list(out.argmax(1).numpy())
        vl, vp, vi = run(iva); vb = balanced_accuracy_score(y[vi], vp)
        log.append(dict(task=task, run=tag, seed=seed, epoch=ep, train_loss=round(tl / len(itr), 4),
                        train_bal=round(balanced_accuracy_score(tt, tp), 4), val_loss=round(vl, 4), val_bal=round(vb, 4)))
        if vb > best:
            best, best_ep, bad = vb, ep, 0; state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= EEGNET["patience"]:
                break
    m.load_state_dict(state)
    pred, ti = (run(te)[1:] if len(te) else (np.array([], dtype=np.int64), te))
    if ckpt:
        torch.save(dict(state_dict=state, task=task, classes=TASKS[task], eegnet=EEGNET, n_params=n_params,
                        best_epoch=best_ep, val_bal_acc=round(best, 4), trained_on=tag,
                        preprocessing="features.raw_input: band-pass 0.5-45 Hz, samples 128:1152 @256 Hz, "
                                      "z-score per channel"), ckpt)
    return dict(tag=tag, seed=seed, te=ti, pred=pred, best_epoch=best_ep, val_bal=best, n_params=n_params, log=log)


# ------------------------------------------------------------------ helpers
def per_recording_z(F, rec):
    """z-score every feature within each recording (labels unused). Removes day/session offsets:
    May-06/07 were exported already band-passed by EmotivPRO, May-08 raw, which shifts absolute band power."""
    F = F.copy()
    for r in np.unique(rec):
        m = rec == r; F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-6)
    return F


def logreg_fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    return sc, LogisticRegression(**LOGREG).fit(sc.transform(Xtr), ytr)


def perm_null(Xtr, ytr, Xte, yte, n, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        sc, lr = logreg_fit(Xtr, rng.permutation(ytr))
        out.append(balanced_accuracy_score(yte, lr.predict(sc.transform(Xte))))
    return np.array(out)


def type_only(ytr, ttr, tte):
    rate = pd.Series(ytr).groupby(ttr).mean()
    return np.array([int(rate.get(t, ytr.mean()) >= ytr.mean()) for t in tte])


def per_type(yt, pr, tt):
    return {f"bal_{t}": round(balanced_accuracy_score(yt[tt == t], pr[tt == t]), 3)
            for t in ["image", "action_image", "video"] if (tt == t).any() and len(np.unique(yt[tt == t])) > 1}


def load(data):
    from datasets import load_dataset, load_from_disk
    ds = load_from_disk(data) if os.path.isdir(data) else load_dataset(data, split="train")
    meta = ds.select_columns(["emotion", "stimulus_type", "eeg_recording_id", "stimulus_onset_unix"]).to_pandas()
    return ds, meta


def cal_split(rec, onset, frac=0.7):
    tr, te = [], []
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]; idx = idx[np.argsort(onset[idx])]; k = int(round(frac * len(idx)))
        tr += list(idx[:k]); te += list(idx[k:])
    return ("CAL", np.array(tr), np.array(te))


def build_configs(day, rec, onset):
    ALL = np.arange(len(day)); last, mid = day == LAST, day == MID
    lo = lambda mask, p: [(f"{p}-{r}", ALL[mask & (rec != r)], ALL[mask & (rec == r)]) for r in sorted(np.unique(rec[mask]))]
    return {
        "A": ("train May-06 + May-07 -> test May-08", [("A", ALL[~last], ALL[last])]),
        "B": ("train May-07 only -> test May-08", [("B", ALL[mid], ALL[last])]),
        "C": ("May-08 only, leave-one-recording-out", lo(last, "C")),
        "D": ("all days, leave-one-recording-out (reference)", lo(np.ones(len(day), bool), "D")),
        "E": ("May-07 + May-08 only, leave-one-recording-out (D without the first day)", lo(last | mid, "E")),
        "CAL": ("calibration: first 70% of every recording (by time) -> train, last 30% -> test", [cal_split(rec, onset)]),
        "ALL": ("train on all three days (final model; its expected accuracy on a new recording is config D)",
                [("ALL", ALL, np.array([], dtype=np.int64))]),
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="valence")
    ap.add_argument("--data", required=True, help="local dataset folder (datasets.load_from_disk)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    ap.add_argument("--configs", default="A,B,C,D,E,CAL,ALL")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        EEGNET.update(max_epochs=3); a.seeds = [1]; a.n_perm = 5
    t0 = time.time(); classes = TASKS[a.task]; n_cls = len(classes)
    out = os.path.join(a.out, a.task); os.makedirs(out, exist_ok=True)

    ds, meta = load(a.data)
    keep = np.where(meta.emotion.isin(POSITIVE | NEGATIVE))[0] if a.task == "valence" else np.arange(len(meta))
    meta = meta.iloc[keep].reset_index(drop=True)
    raw = [np.asarray(ds[int(i)]["eeg_epoch"], dtype=np.float32) for i in keep]
    ctx = get_context("spawn")
    with ctx.Pool(32) as p:
        DE = np.array(p.map(de_features, raw)); XR = np.array(p.map(raw_input, raw))
    y = (meta.emotion.isin(POSITIVE).astype(np.int64).values if a.task == "valence"
         else meta.stimulus_type.map({c: i for i, c in enumerate(classes)}).astype(np.int64).values)
    day = pd.to_datetime(meta.stimulus_onset_unix, unit="s").dt.strftime("%b-%d").values
    rec, typ = meta.eeg_recording_id.values, meta.stimulus_type.values
    DEr = per_recording_z(DE, rec)
    cache = os.path.join(out, "_cache"); np.save(cache + "_X.npy", XR); np.save(cache + "_y.npy", y)
    print(f"[{a.task}] {len(y)} trials, classes {dict(zip(classes, np.bincount(y, minlength=n_cls)))}, "
          f"days {pd.Series(day).value_counts().sort_index().to_dict()}")

    configs = {k: v for k, v in build_configs(day, rec, meta.stimulus_onset_unix.values).items() if k in a.configs.split(",")}
    jobs = []
    for name, (desc, folds) in configs.items():
        d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        json.dump(dict(task=a.task, classes=classes, config=name, description=desc,
                       folds=[dict(tag=t, n_train=len(tr), n_test=len(te),
                                   train_counts=np.bincount(y[tr], minlength=n_cls).tolist(),
                                   test_counts=np.bincount(y[te], minlength=n_cls).tolist()) for t, tr, te in folds],
                       logreg=dict(features="features.de_features: differential entropy, 4 x 1 s windows x 5 bands "
                                            "x 30 channels = 600", **LOGREG),
                       eegnet=dict(input="features.raw_input", epoch_selection="best balanced accuracy on a 15% "
                                   "stratified holdout of the training data", seeds=a.seeds, **EEGNET),
                       controls=[f"{a.n_perm} label permutations (logreg)"] +
                                (["stimulus type only, no EEG"] if a.task == "valence" else [])),
                  open(os.path.join(d, "config.json"), "w"), indent=2)
        for tag, tr, te in folds:
            for s in a.seeds:
                jobs.append((tag, tr, te, s, os.path.join(d, f"eegnet_seed{s}.pt") if name in ("A", "B", "ALL") else None, a.task))

    print(f"EEGNet: {len(jobs)} trainings on CPU")
    with ctx.Pool(min(a.workers, len(jobs)), initializer=_init, initargs=(cache, n_cls)) as p:
        res = p.map(train_eegnet, jobs)
    print(f"  done in {(time.time() - t0) / 60:.1f} min, {res[0]['n_params']} trainable parameters")
    pd.DataFrame([r for e in res for r in e["log"]]).to_csv(os.path.join(out, "training_log.csv"), index=False)

    rows, summary = [], []
    for name, (desc, folds) in configs.items():
        pooled = {}
        nulls, pvals = [], []

        def add(k, te, pr):
            yt, pp, ii = pooled.setdefault(k, ([], [], [])); yt.extend(y[te]); pp.extend(pr); ii.extend(te)

        for tag, tr, te in folds:
            sc, lr = logreg_fit(DEr[tr], y[tr])
            if name == "ALL":
                joblib.dump(dict(scaler=sc, model=lr, task=a.task, classes=classes, trained_on=desc,
                                 features="features.de_features, then z-scored within each recording "
                                          "(models.normalize_session)"), os.path.join(out, name, "logreg.joblib"))
                continue
            p_lr = lr.predict(sc.transform(DEr[te]))
            sc0, lr0 = logreg_fit(DE[tr], y[tr]); p_lr0 = lr0.predict(sc0.transform(DE[te]))
            if name in ("A", "B"):
                joblib.dump(dict(scaler=sc, model=lr, task=a.task, classes=classes, trained_on=desc,
                                 features="features.de_features, then z-scored within each recording "
                                          "(models.normalize_session)"), os.path.join(out, name, "logreg.joblib"))
            real = balanced_accuracy_score(y[te], p_lr)
            if name in "ABC" and len(np.unique(y[te])) > 1:
                nl = perm_null(DEr[tr], y[tr], DEr[te], y[te], a.n_perm)
                nulls.append(nl.mean()); pvals.append((1 + (nl >= real).sum()) / (1 + a.n_perm))
            preds = [("logreg", "", p_lr, te, {}), ("logreg_no_session_norm", "", p_lr0, te, {})]
            if a.task == "valence":
                preds.append(("type_only", "", type_only(y[tr], typ[tr], typ[te]), te, {}))
            preds += [("eegnet", e["seed"], e["pred"], e["te"], dict(best_epoch=e["best_epoch"], val_bal=round(e["val_bal"], 3)))
                      for e in res if e["tag"] == tag]
            for model, seed, pr, ti, extra in preds:
                add(model if model != "eegnet" else f"eegnet_s{seed}", ti, pr)
                rows.append(dict(task=a.task, config=name, fold=tag, model=model, seed=seed, n_train=len(tr), n_test=len(ti),
                                 bal_acc=round(balanced_accuracy_score(y[ti], pr), 3), **extra,
                                 **(per_type(y[ti], pr, typ[ti]) if a.task == "valence" else {})))
        eeg_keys = [k for k in pooled if k.startswith("eegnet_s")]
        for k in ["logreg", "logreg_no_session_norm", "type_only"] + eeg_keys:
            if k not in pooled: continue
            yt, pr, ii = map(np.array, pooled[k])
            row = dict(task=a.task, config=name, description=desc, model=k, n_test=len(yt),
                       bal_acc=round(balanced_accuracy_score(yt, pr), 3),
                       **(per_type(yt, pr, typ[ii]) if a.task == "valence" else {}))
            if name == "D":
                m78 = np.isin(day[ii], [MID, LAST])
                row["bal_testdays_May07_08"] = round(balanced_accuracy_score(yt[m78], pr[m78]), 3)
                for dy in sorted(np.unique(day)):
                    row[f"bal_{dy}"] = round(balanced_accuracy_score(yt[day[ii] == dy], pr[day[ii] == dy]), 3)
            if k == "logreg" and pvals:
                row.update(perm_null_mean=round(float(np.mean(nulls)), 3), perm_p=round(float(max(pvals)), 4))
            summary.append(row)
        if eeg_keys:                                      # mean over EEGNet seeds
            s = [r["bal_acc"] for r in summary if r["config"] == name and r["model"].startswith("eegnet_s")]
            summary.append(dict(task=a.task, config=name, description=desc, model="eegnet_mean_seeds",
                                bal_acc=round(float(np.mean(s)), 3), bal_acc_sd=round(float(np.std(s)), 3)))

    pd.DataFrame(rows).to_csv(os.path.join(out, "results.csv"), index=False)
    S = pd.DataFrame(summary); S.to_csv(os.path.join(out, "summary.csv"), index=False)
    for f in ("_cache_X.npy", "_cache_y.npy"):
        os.remove(os.path.join(out, f))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(S.drop(columns=["description"]).fillna("").to_string(index=False))
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
