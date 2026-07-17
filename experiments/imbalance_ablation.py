# -*- coding: utf-8 -*-
"""Imbalance-strategy ablation (O1 justification).

Holds the model fixed (XGBoost, the strongest baseline) and varies only the
imbalance-handling strategy, so the effect is isolated:
    none | class-weight(balanced) | class-weight(sqrt) | SMOTE(train-only)
Reports overall imbalance-aware metrics and the rare-class recall/precision
trade-off on the same held-out test set.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib

import models as M
import metrics as MET

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
os.makedirs(TAB, exist_ok=True)
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xtr, ytr, Xte, yte = L("X_train"), L("y_train"), L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
RARE = ["BruteForce", "WebAttack", "Bot"]
print(f"train {Xtr.shape}  test {Xte.shape}")

def fit_eval(sample_weight=None, resample=None):
    Xt, yt = (resample if resample else (Xtr, ytr))
    xgb = M.build_xgboost(num_classes=len(cn))
    xgb.fit(Xt, yt, sample_weight=sample_weight)
    return MET.evaluate(yte, xgb.predict(Xte), cn, y_score=xgb.predict_proba(Xte))

runs = {}
t = time.time()
runs["none"] = fit_eval(sample_weight=None)
print(f"none done ({time.time()-t:.0f}s)"); t = time.time()
runs["weight-balanced"] = fit_eval(sample_weight=M.balanced_sample_weight(ytr, "balanced"))
print(f"balanced done ({time.time()-t:.0f}s)"); t = time.time()
runs["weight-sqrt"] = fit_eval(sample_weight=M.balanced_sample_weight(ytr, "sqrt"))
print(f"sqrt done ({time.time()-t:.0f}s)"); t = time.time()

# SMOTE: train-only, moderate oversampling of rare classes (not full parity)
try:
    from imblearn.over_sampling import SMOTE
    idx = {c: i for i, c in enumerate(cn)}
    target = {idx[c]: 25000 for c in ["Infiltration"] + RARE}  # only bump the sparse ones
    sm = SMOTE(sampling_strategy=target, k_neighbors=5, random_state=42)
    Xr, yr = sm.fit_resample(Xtr, ytr)
    print(f"SMOTE: train {len(ytr):,} -> {len(yr):,}")
    runs["SMOTE(train-only)"] = fit_eval(resample=(Xr, yr))
    print(f"SMOTE done ({time.time()-t:.0f}s)")
except Exception as e:
    print("SMOTE skipped:", e)

# ---- overall table ----
overall = pd.DataFrame({k: v["overall"] for k, v in runs.items()}).T[
    ["accuracy", "macro_f1", "balanced_accuracy", "macro_fpr", "weighted_f1"]].round(4)
overall.index.name = "strategy"
overall.to_csv(os.path.join(TAB, "imbalance_ablation_overall.csv"))
print("\n=== overall (test set) ===\n", overall.to_string())

# ---- rare-class recall & precision trade-off ----
rows = {}
for k, v in runs.items():
    pc = v["per_class"].set_index("class")
    row = {}
    for c in RARE:
        row[f"{c}_recall"] = round(pc.loc[c, "recall"], 3)
        row[f"{c}_prec"] = round(pc.loc[c, "precision"], 3)
    rows[k] = row
rare = pd.DataFrame(rows).T
rare.index.name = "strategy"
rare.to_csv(os.path.join(TAB, "imbalance_ablation_rare.csv"))
print("\n=== rare-class recall / precision ===\n", rare.to_string())
print("\nsaved -> results/tables/imbalance_ablation_*.csv")
