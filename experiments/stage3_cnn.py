# -*- coding: utf-8 -*-
"""Stage 3a (O2) — develop the primary 1D-CNN detector on the improved CICIDS2017.

Trains the class-weighted 1D-CNN on the cached leakage-free splits, evaluates it
on the held-out test set with the same imbalance-aware metrics as the baselines,
saves the model, and prints a baseline-vs-CNN comparison. FGSM/PGD attacks (O2)
and adversarial training (O3) build on the model saved here.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib

import models as M
import metrics as MET

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
os.makedirs(MODELS, exist_ok=True)
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xtr, ytr = L("X_train"), L("y_train")
Xva, yva = L("X_val"), L("y_val")
Xte, yte = L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  classes {len(cn)}")

t = time.time()
cnn = M.train_cnn1d(Xtr, ytr, Xva, yva, n_classes=len(cn), max_epochs=30,
                    patience=6, weight_scheme="balanced", verbose=True)
print(f"1D-CNN trained in {time.time()-t:.0f}s")

res = MET.evaluate(yte, cnn.predict(Xte), cn, y_score=cnn.predict_proba(Xte))
o = res["overall"]
print(f"\n1D-CNN test: acc={o['accuracy']:.4f}  macroF1={o['macro_f1']:.4f}  "
      f"bal_acc={o['balanced_accuracy']:.4f}  MCC={o['mcc']:.4f}  macroFPR={o['macro_fpr']:.4f}")
print("\nper-class:\n", res["per_class"].round(3).to_string(index=False))

M.save_cnn(cnn, os.path.join(MODELS, "cnn.pt"), n_classes=len(cn))
res["per_class"].to_csv(os.path.join(TAB, "cnn_per_class.csv"), index=False)

# append CNN to the baseline comparison
row = {"model": "1D-CNN", **o}
try:
    base = pd.read_csv(os.path.join(TAB, "baselines_summary.csv"))
    allm = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
except Exception:
    allm = pd.DataFrame([row])
keep = ["model", "accuracy", "macro_f1", "balanced_accuracy", "mcc", "macro_fpr", "macro_pr_auc"]
allm = allm[[c for c in keep if c in allm.columns]].sort_values("macro_f1", ascending=False)
allm.to_csv(os.path.join(TAB, "models_summary.csv"), index=False)
print("\n=== all models (test) ===\n", allm.round(4).to_string(index=False))
print("\nsaved -> results/models/cnn.pt  &  results/tables/cnn_per_class.csv, models_summary.csv")
