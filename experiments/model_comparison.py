# -*- coding: utf-8 -*-
"""O1 — comprehensive multi-model comparison.

Trains many detector families on ONE consistent training subsample (per-class
capped for tractability) and evaluates them all on the FULL held-out test set
with the same imbalance-aware metrics, so no single model is assumed.

Results are written incrementally to results/tables/model_comparison.csv — the
script is RESUMABLE (already-finished models are skipped), so a session
interruption never loses completed work.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib

import models as M
import models_extra as MX
import metrics as MET

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
os.makedirs(TAB, exist_ok=True)
CSV = os.path.join(TAB, "model_comparison.csv")
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xtr, ytr = L("X_train"), L("y_train")
Xva, yva = L("X_val"), L("y_val")
Xte, yte = L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
nf, nc = Xtr.shape[1], len(cn)

# consistent per-class-capped train + val (rare classes kept fully) for a fair,
# tractable sweep; final evaluation is on the FULL test set.
def cap(X, y, per_class, seed=42):
    rng = np.random.RandomState(seed); keep = []
    for c in range(nc):
        idx = np.where(y == c)[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep); rng.shuffle(keep)
    return X[keep], y[keep]

Xtr, ytr = cap(Xtr, ytr, 25000)
Xvs, yvs = cap(Xva, yva, 5000)          # small val for fast early-stopping
print(f"sweep train {Xtr.shape}  val {Xvs.shape}  test {Xte.shape}  classes {nc}", flush=True)

def _xgb():
    m = M.build_xgboost(num_classes=nc)
    return m.fit(Xtr, ytr, sample_weight=M.balanced_sample_weight(ytr, "balanced"))

MODELS = {
    # classical
    "RandomForest":       lambda: M.build_random_forest().fit(Xtr, ytr),
    "XGBoost":            _xgb,
    "DecisionTree":       lambda: MX.build_decision_tree().fit(Xtr, ytr),
    "LogisticRegression": lambda: MX.build_logistic().fit(Xtr, ytr),
    # deep
    "MLP":       lambda: M.build_mlp().fit(Xtr, ytr),
    "DNN":       lambda: MX.train_net(MX.build_dnn_net(nf, nc), Xtr, ytr, Xvs, yvs, nc, tag="DNN"),
    "1D-CNN":    lambda: MX.train_net(M._build_cnn1d_net(nf, nc), Xtr, ytr, Xvs, yvs, nc, tag="CNN"),
    "LSTM":      lambda: MX.train_net(MX.build_lstm_net(nf, nc), Xtr, ytr, Xvs, yvs, nc, tag="LSTM"),
    "GRU":       lambda: MX.train_net(MX.build_gru_net(nf, nc), Xtr, ytr, Xvs, yvs, nc, tag="GRU"),
    "CNN-LSTM":  lambda: MX.train_net(MX.build_cnn_lstm_net(nf, nc), Xtr, ytr, Xvs, yvs, nc, tag="CNNLSTM"),
    "TabNet":    lambda: M.train_tabnet(Xtr, ytr, Xvs, yvs, max_epochs=40, patience=8),
}

done = set(pd.read_csv(CSV)["model"]) if os.path.exists(CSV) else set()
print(f"already done: {sorted(done)}", flush=True)

for name, fn in MODELS.items():
    if name in done:
        print(f"[skip] {name}", flush=True); continue
    try:
        t = time.time(); model = fn(); dt = time.time() - t
        ys = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
        res = MET.evaluate(yte, model.predict(Xte), cn, y_score=ys)
        o = res["overall"]; row = {"model": name, **o, "train_time_s": round(dt, 1)}
        # append to the running CSV (resumable)
        df = pd.concat([pd.read_csv(CSV), pd.DataFrame([row])], ignore_index=True) \
             if os.path.exists(CSV) else pd.DataFrame([row])
        df.to_csv(CSV, index=False)
        res["per_class"].to_csv(os.path.join(TAB, f"cmp_perclass_{name}.csv"), index=False)
        print(f"[done] {name:<18} {dt:6.0f}s  macroF1={o['macro_f1']:.4f}  "
              f"balAcc={o['balanced_accuracy']:.4f}  MCC={o['mcc']:.4f}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {e}", flush=True)

# final ranked table
allm = pd.read_csv(CSV).sort_values("macro_f1", ascending=False)
print("\n=== comprehensive comparison (test set), ranked by macro-F1 ===")
cols = ["model", "accuracy", "macro_f1", "balanced_accuracy", "mcc", "macro_fpr", "macro_pr_auc", "train_time_s"]
print(allm[[c for c in cols if c in allm.columns]].round(4).to_string(index=False))
