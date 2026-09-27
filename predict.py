"""
Run the saved checkpoints on new trials.

Input: trials from ONE recording session, as a .npy array of shape (n_trials, 30, 1408):
30 channels in the Emotiv FLEX2 order used for training, 256 Hz, from -0.5 s to +5.0 s around stimulus
onset (the `eeg_epoch` format). With access to the dataset, --recording pulls a whole recording.

The logistic-regression checkpoint expects its features z-scored within the session, so pass
a whole session (at least ~30 trials), not single trials. EEGNet works per trial; its output
is averaged over the saved seeds.

Examples
  python predict.py --ckpt checkpoints/stimtype --input my_session.npy
  python predict.py --ckpt checkpoints/stimtype --recording <id> --data /path/to/dataset
"""
import argparse, glob, json, os, numpy as np, pandas as pd

from features import de_features, raw_input
from models import load_eegnet, load_logreg, eegnet_proba, logreg_proba, normalize_session


def load_recording(data, rec):
    from datasets import load_dataset, load_from_disk
    ds = load_from_disk(data) if os.path.isdir(data) else load_dataset(data, split="train")
    ids = [i for i, r in enumerate(ds["eeg_recording_id"]) if r == rec]
    if not ids:
        raise SystemExit(f"no trials for recording {rec}")
    sub = ds.select(ids)
    X = np.stack([np.asarray(e, dtype=np.float32) for e in sub["eeg_epoch"]])
    meta = sub.select_columns(["trial_id", "stimulus_type", "emotion", "stimulus_filename"]).to_pandas()
    return X, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="folder with logreg.joblib and eegnet_seed*.pt")
    ap.add_argument("--input", help=".npy of shape (n, 30, 1408)")
    ap.add_argument("--recording", help="eeg_recording_id to pull from --data")
    ap.add_argument("--data", help="local dataset folder, only needed with --recording")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()

    if a.input:
        X, meta = np.load(a.input), pd.DataFrame()
    elif a.recording:
        X, meta = load_recording(a.data, a.recording)
    else:
        raise SystemExit("give --input or --recording")
    if len(X) < 30:
        print(f"warning: only {len(X)} trials; session normalisation for the logistic regression is noisy")

    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    classes = cfg["classes"]
    out = meta.copy()

    lr = load_logreg(os.path.join(a.ckpt, "logreg.joblib"))
    p_lr = logreg_proba(lr, normalize_session(np.stack([de_features(e) for e in X])))
    R = np.stack([raw_input(e) for e in X])
    seeds = sorted(glob.glob(os.path.join(a.ckpt, "eegnet_seed*.pt")))
    p_eeg = np.mean([eegnet_proba(load_eegnet(s)[0], R) for s in seeds], axis=0)

    for name, P in [("logreg", p_lr), ("eegnet", p_eeg)]:
        out[f"{name}_pred"] = [classes[k] for k in P.argmax(1)]
        for k, c in enumerate(classes):
            out[f"{name}_p_{c}"] = P[:, k].round(4)
    out.to_csv(a.out, index=False)
    print(f"{cfg['task']} | {len(X)} trials | EEGNet seeds: {len(seeds)} | wrote {a.out}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
