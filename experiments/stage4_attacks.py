# -*- coding: utf-8 -*-
"""Stage 4 (O2) — adversarial evasion of the 1D-CNN.

Attacks the trained CNN with a random baseline (Gaussian) and true white-box
gradient attacks (FGSM, PGD) across increasing perturbation strength epsilon,
reporting the attack->Benign false-negative rate and overall evasion rate. Also
reports which attack classes are breached first at eps=0.1 (PGD).
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib

import models as M
import security as S

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
FIG = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xte, yte = L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
BENIGN = cn.index("Benign")

cnn = M.load_cnn(os.path.join(MODELS, "cnn.pt"))

# eval set: per-class capped attack sample (full rare-class coverage) + benign
# sample — keeps the gradient attacks tractable on CPU.
rng = np.random.RandomState(0)
parts = []
for c in range(len(cn)):
    idx_c = np.where(yte == c)[0]
    cap = 15000 if c == BENIGN else 2000
    if len(idx_c) > cap:
        idx_c = rng.choice(idx_c, cap, replace=False)
    parts.append(idx_c)
sel = np.concatenate(parts)
Xs, ys = Xte[sel].astype("float32"), yte[sel]
print(f"attack-eval set: {len(ys):,} flows ({int((ys != BENIGN).sum()):,} attack + "
      f"{int((ys == BENIGN).sum()):,} benign)")

EPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
rows = []
for scheme in ["gaussian", "fgsm_grad", "pgd_grad"]:
    t = time.time()
    rc = S.robustness_curve(cnn, Xs, ys, EPS, scheme=scheme,
                            n_classes=len(cn), benign_index=BENIGN)
    for i, e in enumerate(EPS):
        rows.append({"scheme": scheme, "epsilon": e,
                     "accuracy": rc["accuracy"][i], "macro_f1": rc["macro_f1"][i],
                     "attack_to_benign_fnr": rc["attack_to_benign_fnr"][i],
                     "evasion_rate": rc["evasion_rate"][i]})
    print(f"{scheme:<10} done ({time.time()-t:.0f}s)  "
          f"evasion@0.1={rc['evasion_rate'][EPS.index(0.1)]:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(TAB, "cnn_robustness.csv"), index=False)
print("\n=== evasion_rate by scheme x epsilon ===")
print(df.pivot(index="epsilon", columns="scheme", values="evasion_rate").round(3).to_string())

# per-class breach at eps=0.1 under PGD
pcf = S.per_class_fnr(cnn, Xs, ys, 0.1, cn, scheme="pgd_grad", n_classes=len(cn))
pcf = pd.Series(pcf, name="pgd_fnr@0.1").rename_axis("class")
pcf.to_csv(os.path.join(TAB, "cnn_perclass_evasion_pgd01.csv"))
print("\n=== per-class FNR under PGD (eps=0.1) ===\n", pcf.round(3).to_string())

# figure: evasion vs epsilon
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for scheme, c, lbl in [("gaussian", "#999999", "Gaussian (random)"),
                       ("fgsm_grad", "#E1812C", "FGSM (gradient)"),
                       ("pgd_grad", "#C62828", "PGD (gradient)")]:
    d = df[df.scheme == scheme]
    ax.plot(d["epsilon"], d["attack_to_benign_fnr"], marker="o", color=c, label=lbl)
ax.set_xlabel("perturbation strength  ε  (fraction of feature IQR)")
ax.set_ylabel("attack → Benign FNR (evasion)")
ax.set_title("1D-CNN adversarial vulnerability (O2)")
ax.grid(alpha=0.3); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cnn_attacks.png"), dpi=150)
print("\nsaved -> results/tables/cnn_robustness.csv, cnn_perclass_evasion_pgd01.csv & figure")
