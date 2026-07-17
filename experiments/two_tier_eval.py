# -*- coding: utf-8 -*-
"""Evaluate the two-tier XGBoost + hardened-DNN system: clean performance and
adversarial robustness (defence in depth). Attacks are crafted with the DNN's
gradient (PGD); we show that attacks evading the DNN are still caught by the
non-differentiable XGBoost tier, so evading the two-tier requires fooling both."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib
import metrics as MET
import security as S
import two_tier

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xte, yte = L("X_test"), L("y_test")
det = two_tier.TwoTierDetector(MODELS, PROC)
cn = det.class_names; BEN = det.benign
xgb, dnn = det.xgb, det.dnn

# ---- clean performance ----
print("=== clean performance (full test) ===")
rows = []
for name, pred in [("XGBoost", xgb.predict(Xte)),
                   ("hardened-DNN", dnn.predict(Xte)),
                   ("two-tier", det.predict(Xte))]:
    o = MET.evaluate(yte, pred, cn)["overall"]
    rows.append({"model": name, "macro_f1": round(o["macro_f1"], 4),
                 "balanced_acc": round(o["balanced_accuracy"], 4),
                 "macro_fpr": round(o["macro_fpr"], 4)})
clean = pd.DataFrame(rows); print(clean.to_string(index=False))

# ---- robustness under PGD crafted on the DNN ----
rng = np.random.RandomState(0); parts = []
for c in range(len(cn)):
    idx = np.where(yte == c)[0]
    cap = 15000 if c == BEN else 2000
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    parts.append(idx)
sel = np.concatenate(parts); Xs, ys = Xte[sel].astype("float32"), yte[sel]
atk = ys != BEN
EPS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
print(f"\n=== robustness: PGD crafted on the DNN, {int(atk.sum()):,} attack flows ===")

rob = []
for e in EPS:
    Xadv = Xs if e == 0 else S.pgd_grad(dnn, Xs, ys, e)
    dnn_p = dnn.predict(Xadv)[atk]
    xgb_p = xgb.predict(Xadv)[atk]
    ev_dnn = float((dnn_p == BEN).mean())
    ev_xgb = float((xgb_p == BEN).mean())
    ev_two = float(((dnn_p == BEN) & (xgb_p == BEN)).mean())   # evades BOTH tiers
    rob.append({"epsilon": e, "evasion_DNN_only": round(ev_dnn, 3),
                "evasion_XGB_on_adv": round(ev_xgb, 3), "evasion_two_tier": round(ev_two, 3)})
    print(f"  eps={e:<4}  DNN-only={ev_dnn:.3f}  XGB-on-adv={ev_xgb:.3f}  two-tier={ev_two:.3f}", flush=True)

rob = pd.DataFrame(rob)
clean.to_csv(os.path.join(TAB, "two_tier_clean.csv"), index=False)
rob.to_csv(os.path.join(TAB, "two_tier_robustness.csv"), index=False)
print("\nsaved -> results/tables/two_tier_clean.csv, two_tier_robustness.csv")
