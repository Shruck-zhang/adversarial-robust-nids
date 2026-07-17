"""Data loading, label engineering and cleaning for the improved CICIDS2017
(Engelen, Rimmer & Joosen, WTMC 2021).

The improved dataset differs from the standard MachineLearningCVE release:
  * 5 day-files (monday..friday), 91 columns, ~2.10M flows;
  * identifier columns (id, Flow ID, Src/Dst IP, Src Port, Timestamp) are present
    and are DROPPED for modelling (kept only for attribution);
  * new features: Fwd/Bwd RST Flags, ICMP Code/Type, Total TCP Flow Time;
  * 27 fine-grained labels including `- Attempted` variants (attacks that were
    initiated but carried no payload).

We map the 27 labels onto an 8-class taxonomy and record an `is_attempted` flag.
The Attempted handling is configurable (see `attempted_policy`).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

LABEL_COL = "Label"

# Columns dropped before modelling: pure identifiers + leakage risks + the extra
# attempted-category metadata. Dst Port and Protocol are retained as features.
ID_COLS = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp",
           "Attempted Category"]

CLASS_NAMES = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce",
               "WebAttack", "Bot", "Infiltration"]

# Base-label (attempted suffix already stripped) -> analysis class, matched as a
# priority-ordered substring test on the upper-cased label.
DROPPED_BASE = {"HEARTBLEED"}  # 11 rows, too few to learn/evaluate


def map_label(raw: str, attempted_policy: str = "merge"):
    """Map an improved-CICIDS2017 label to (class, is_attempted).

    attempted_policy:
      'merge'  -> an attempted attack keeps its parent attack class (default)
      'benign' -> an attempted (payload-less) attack is treated as Benign
      'drop'   -> attempted rows are dropped (class = None)
    Returns (class_name_or_None, is_attempted_bool).
    """
    s = str(raw).strip()
    low = s.lower()
    is_attempted = "attempted" in low
    if is_attempted:
        if attempted_policy == "drop":
            return None, True
        if attempted_policy == "benign":
            return "Benign", True
    up = s.upper()
    if "BENIGN" in up:
        return "Benign", is_attempted
    # order matters: infiltration before portscan; ddos before dos; bot via 'BOT'
    if "INFILTRATION" in up:
        return "Infiltration", is_attempted
    if "PORTSCAN" in up or "PORT SCAN" in up:
        return "PortScan", is_attempted
    if "DDOS" in up:
        return "DDoS", is_attempted
    if "DOS" in up:
        return "DoS", is_attempted
    if "PATATOR" in up:
        return "BruteForce", is_attempted
    if "WEB ATTACK" in up or "WEBATTACK" in up:
        return "WebAttack", is_attempted
    if "BOT" in up:
        return "Bot", is_attempted
    if any(d in up for d in DROPPED_BASE):
        return None, is_attempted
    return None, is_attempted


def load_clean_dataset(attempted_policy="merge", raw_dir=RAW_DIR, verbose=True):
    """Load all day-files, map labels, drop identifiers, coerce numeric, remove
    non-finite and exact-duplicate flows. Returns (X_df, y_series, is_attempted)."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs in {raw_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        if verbose:
            print(f"  loaded {os.path.basename(f)}: {len(df):,} rows")
    df = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"concatenated: {len(df):,} rows, {df.shape[1]} cols")

    mapped = df[LABEL_COL].map(lambda r: map_label(r, attempted_policy))
    y = mapped.map(lambda t: t[0])
    is_att = mapped.map(lambda t: t[1])
    keep = y.notna()
    df, y, is_att = df[keep], y[keep], is_att[keep]
    if verbose:
        print(f"after label mapping/drop: {len(df):,} rows ({CLASS_NAMES})")

    drop = [c for c in ID_COLS + [LABEL_COL] if c in df.columns]
    X = df.drop(columns=drop)
    # coerce every feature to numeric; inf -> nan -> drop those rows
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = X.notna().all(axis=1)
    X, y, is_att = X[finite], y[finite], is_att[finite]
    if verbose:
        print(f"after numeric coercion + finite filter: {len(X):,} rows, {X.shape[1]} features")

    # exact-duplicate flows (feature-identical) removed to prevent train/test leakage
    dedup = ~X.duplicated()
    X, y, is_att = X[dedup].reset_index(drop=True), y[dedup].reset_index(drop=True), is_att[dedup].reset_index(drop=True)
    if verbose:
        print(f"after exact-duplicate removal: {len(X):,} rows")
    return X, y.astype(str), is_att.astype(bool)


def encode_labels(y):
    """Map class-name strings to fixed integer ids (CLASS_NAMES order)."""
    idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    return np.array([idx[v] for v in y], dtype=int)


def make_splits(X, y, is_att=None, test_size=0.2, val_size=0.2, clip=10.0,
                group_sig=2, corr_threshold=0.98, random_state=42, verbose=True,
                return_raw=False):
    """Leakage-free 60/20/20 split + RobustScaler + train-only feature pruning.

    Near-identical flows (rounded to `group_sig` significant figures) are kept on
    one side of every split via StratifiedGroupKFold, so bursts of near-twin
    attack flows cannot leak between train and test. Scaling and pruning are fit
    on the training split only.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import RobustScaler

    feat = list(X.columns)
    Xv = X.to_numpy("float64")
    yi = encode_labels(y)
    att = np.zeros(len(yi), bool) if is_att is None else np.asarray(is_att, bool)

    # group key: signed 2-sig-fig fingerprint of each flow
    def sigfig(a, s):
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = np.floor(np.log10(np.abs(a)))
            factor = 10.0 ** (s - 1 - mag)
            r = np.round(a * factor) / factor
        return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    groups = pd.util.hash_pandas_object(
        pd.DataFrame(sigfig(Xv, group_sig)).round(6), index=False).to_numpy()

    def split_once(Xa, ya, ga, aa, frac, seed):
        n_splits = max(2, int(round(1 / frac)))
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, te = next(skf.split(Xa, ya, groups=ga))
        return tr, te

    idx_all = np.arange(len(yi))
    tr_idx, te_idx = split_once(Xv, yi, groups, att, test_size, random_state)
    Xtr0, ytr0, gtr0, atr0 = Xv[tr_idx], yi[tr_idx], groups[tr_idx], att[tr_idx]
    val_frac = val_size / (1 - test_size)
    tr2, va2 = split_once(Xtr0, ytr0, gtr0, atr0, val_frac, random_state + 1)

    Xtr, ytr, atr = Xtr0[tr2], ytr0[tr2], atr0[tr2]
    Xva, yva, ava = Xtr0[va2], ytr0[va2], atr0[va2]
    Xte, yte, ate = Xv[te_idx], yi[te_idx], att[te_idx]

    # train-only feature pruning: drop zero-variance + one of each corr>thr pair
    std = Xtr.std(axis=0)
    keep = std > 1e-12
    Xtr, Xva, Xte = Xtr[:, keep], Xva[:, keep], Xte[:, keep]
    feat = [f for f, k in zip(feat, keep) if k]
    if corr_threshold:
        corr = np.abs(np.corrcoef(Xtr, rowvar=False))
        drop = set()
        for i in range(len(feat)):
            if i in drop:
                continue
            for j in range(i + 1, len(feat)):
                if j not in drop and corr[i, j] > corr_threshold:
                    drop.add(j)
        keep2 = [i for i in range(len(feat)) if i not in drop]
        Xtr, Xva, Xte = Xtr[:, keep2], Xva[:, keep2], Xte[:, keep2]
        feat = [feat[i] for i in keep2]

    scaler = RobustScaler().fit(Xtr)
    Xtr_raw, Xva_raw, Xte_raw = Xtr.copy(), Xva.copy(), Xte.copy()  # pre-scale physical values
    def scale(a):
        return np.clip(scaler.transform(a), -clip, clip).astype("float32")
    Xtr, Xva, Xte = scale(Xtr), scale(Xva), scale(Xte)

    if verbose:
        print(f"split: train {Xtr.shape} val {Xva.shape} test {Xte.shape}  "
              f"features {len(feat)}")
    out = {
        "X_train": Xtr, "y_train": ytr, "att_train": atr,
        "X_val": Xva, "y_val": yva, "att_val": ava,
        "X_test": Xte, "y_test": yte, "att_test": ate,
        "scaler": scaler, "feature_names": feat, "class_names": CLASS_NAMES,
    }
    if return_raw:
        out["X_train_raw"] = Xtr_raw.astype("float32")
        out["X_val_raw"] = Xva_raw.astype("float32")
        out["X_test_raw"] = Xte_raw.astype("float32")
    return out
