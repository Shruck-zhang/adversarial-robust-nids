# -*- coding: utf-8 -*-
"""Stage 2 — baseline detectors on the improved CICIDS2017.

Trains Random Forest, XGBoost and an MLP/DNN on the cached leakage-free splits
and reports per-class + imbalance-aware metrics (O1). Results are written to
results/tables/.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib

import models as M
import metrics as MET

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
MODELS_OUT = os.path.join(os.path.dirname(__file__), "..", "results", "models")
os.makedirs(TAB, exist_ok=True); os.makedirs(MODELS_OUT, exist_ok=True)

L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xtr, ytr = L("X_train"), L("y_train")
Xte, yte = L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
print(f"train {Xtr.shape}  test {Xte.shape}  classes {len(cn)}")

results, fitted, timings = {}, {}, {}

def run(name, fit_fn):
    t = time.time()
    model = fit_fn()
    timings[name] = time.time() - t
    yp = model.predict(Xte)
    ys = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
    results[name] = MET.evaluate(yte, yp, cn, y_score=ys)
    fitted[name] = model
    o = results[name]["overall"]
    print(f"{name:<14} {timings[name]:6.0f}s  acc={o['accuracy']:.4f}  "
          f"macroF1={o['macro_f1']:.4f}  bal_acc={o['balanced_accuracy']:.4f}  "
          f"MCC={o['mcc']:.4f}  macroFPR={o['macro_fpr']:.4f}")

# --- Random Forest (balanced class weights) ---
run("RandomForest", lambda: M.build_random_forest().fit(Xtr, ytr))
# --- XGBoost (inverse-frequency per-sample weights) ---
def fit_xgb():
    xgb = M.build_xgboost(num_classes=len(cn))
    return xgb.fit(Xtr, ytr, sample_weight=M.balanced_sample_weight(ytr, "balanced"))
run("XGBoost", fit_xgb)
# --- MLP / DNN (main deep baseline) ---
run("MLP", lambda: M.build_mlp().fit(Xtr, ytr))

# ---- save tables ----
summary = MET.comparison_table(results)
summary["train_time_s"] = summary["model"].map(lambda m: round(timings[m], 1))
summary.to_csv(os.path.join(TAB, "baselines_summary.csv"), index=False)
print("\n=== overall (test set) ===")
print(summary.to_string(index=False))

# per-class recall matrix (rows=class, cols=model)
rec = pd.DataFrame({m: results[m]["per_class"].set_index("class")["recall"] for m in results})
rec.to_csv(os.path.join(TAB, "baselines_per_class_recall.csv"))
print("\n=== per-class recall (detection rate) ===")
print(rec.round(3).to_string())

for m in results:
    results[m]["per_class"].to_csv(os.path.join(TAB, f"baselines_per_class_{m}.csv"), index=False)
    joblib.dump(fitted[m], os.path.join(MODELS_OUT, f"{m}.pkl"))

print("\nsaved -> results/tables/baselines_*.csv  &  results/models/*.pkl")
