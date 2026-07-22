# -*- coding: utf-8 -*-
"""Repeated-run confidence intervals across random seeds.

Trains the key architectures under N random seeds on the same per-class-capped
training budget and evaluates each on the FULL held-out test set, then reports the
mean and a 95% t-based confidence interval of macro-F1. Resumable: each (model,seed)
row is appended to a CSV and re-runs skip completed cells."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib
import models as M, models_extra as MX, metrics as MET

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xtr0, ytr0 = L("X_train"), L("y_train")
Xva0, yva0 = L("X_val"), L("y_val")
Xte, yte = L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
nf, nc = Xtr0.shape[1], len(cn)
ROWS = os.path.join(TAB, "multiseed_ci_rows.csv")
SEEDS = [0, 1, 2, 3, 4]

def cap(X, y, per, seed):
    rng = np.random.RandomState(seed); keep = []
    for c in range(nc):
        idx = np.where(y == c)[0]
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        keep.append(idx)
    s = np.concatenate(keep)
    return X[s], y[s]

def fit_eval(model, seed):
    Xtr, ytr = cap(Xtr0, ytr0, 25000, seed)
    Xvs, yvs = cap(Xva0, yva0, 5000, seed)
    if model == "XGBoost":
        m = M.build_xgboost(num_classes=nc, random_state=seed)
        m.fit(Xtr, ytr, sample_weight=M.balanced_sample_weight(ytr, "balanced")); pred = m.predict(Xte)
    elif model == "RandomForest":
        m = M.build_random_forest(random_state=seed).fit(Xtr, ytr); pred = m.predict(Xte)
    elif model == "MLP (sklearn)":
        m = M.build_mlp(random_state=seed).fit(Xtr, ytr); pred = m.predict(Xte)
    elif model == "DNN":
        w = MX.train_net(MX.build_dnn_net(nf, nc), Xtr, ytr, Xvs, yvs, nc, random_state=seed, tag="DNN")
        pred = w.predict(Xte)
    elif model == "1D-CNN":
        w = MX.train_net(M._build_cnn1d_net(nf, nc), Xtr, ytr, Xvs, yvs, nc, random_state=seed, tag="CNN")
        pred = w.predict(Xte)
    return MET.evaluate(yte, pred, cn)["overall"]["macro_f1"]

MODELS = ["XGBoost", "RandomForest", "MLP (sklearn)", "DNN", "1D-CNN"]
done = set()
if os.path.exists(ROWS):
    d = pd.read_csv(ROWS)
    done = {(r.model, int(r.seed)) for r in d.itertuples()}

for model in MODELS:
    for seed in SEEDS:
        if (model, seed) in done:
            continue
        t = time.time(); f1 = fit_eval(model, seed)
        pd.DataFrame([{"model": model, "seed": seed, "macro_f1": round(f1, 4)}]).to_csv(
            ROWS, mode="a", header=not os.path.exists(ROWS), index=False)
        print(f"  {model:14s} seed={seed}  macroF1={f1:.4f}  ({time.time()-t:.0f}s)", flush=True)

# aggregate -> mean + 95% t-CI
d = pd.read_csv(ROWS)
from scipy import stats
out = []
for model in MODELS:
    v = d[d.model == model]["macro_f1"].values
    if len(v) < 2:
        continue
    m, sd, n = v.mean(), v.std(ddof=1), len(v)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    out.append({"model": model, "n_seeds": n, "macro_f1_mean": round(m, 4),
                "std": round(sd, 4), "ci95_halfwidth": round(h, 4),
                "ci_low": round(m - h, 4), "ci_high": round(m + h, 4)})
agg = pd.DataFrame(out)
agg.to_csv(os.path.join(TAB, "multiseed_ci.csv"), index=False)
print("\n=== macro-F1 over seeds (mean ± 95% CI) ===")
print(agg.to_string(index=False))
print("\nsaved -> results/tables/multiseed_ci.csv")
