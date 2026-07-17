# -*- coding: utf-8 -*-
"""Stage 3b (O3) — adversarial training (defence) of the 1D-CNN.

Adversarially trains the CNN (per-batch FGSM), evaluates the clean-accuracy cost
and the robustness gain under FGSM/PGD, and reports the before/after
robustness-accuracy trade-off vs the undefended CNN from stage3_cnn.py.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib

import models as M
import metrics as MET
import security as S

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xtr, ytr = L("X_train"), L("y_train")
Xva, yva = L("X_val"), L("y_val")
Xte, yte = L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
BEN = cn.index("Benign")

t = time.time()
hard = M.train_cnn1d_adversarial(Xtr, ytr, Xva, yva, n_classes=len(cn),
                                 max_epochs=20, patience=6, eps=0.1,
                                 weight_scheme="balanced", verbose=True)
print(f"hardened CNN trained in {time.time()-t:.0f}s")
M.save_cnn(hard, os.path.join(MODELS, "cnn_hardened.pt"), n_classes=len(cn))

plain = M.load_cnn(os.path.join(MODELS, "cnn.pt"))

# clean cost
rp = MET.evaluate(yte, plain.predict(Xte), cn)["overall"]
rh = MET.evaluate(yte, hard.predict(Xte), cn)["overall"]
print(f"\nclean macro-F1: plain={rp['macro_f1']:.4f}  hardened={rh['macro_f1']:.4f}  "
      f"(cost {rp['macro_f1']-rh['macro_f1']:+.4f})")

# robustness before/after on a per-class capped eval set (tractable on CPU)
rng = np.random.RandomState(0)
parts = []
for c in range(len(cn)):
    idx_c = np.where(yte == c)[0]
    cap = 15000 if c == BEN else 2000
    if len(idx_c) > cap:
        idx_c = rng.choice(idx_c, cap, replace=False)
    parts.append(idx_c)
sel = np.concatenate(parts); Xs, ys = Xte[sel].astype("float32"), yte[sel]
EPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

rows = []
for tag, mdl in [("plain", plain), ("hardened", hard)]:
    for scheme in ["fgsm_grad", "pgd_grad"]:
        rc = S.robustness_curve(mdl, Xs, ys, EPS, scheme=scheme, n_classes=len(cn), benign_index=BEN)
        for i, e in enumerate(EPS):
            rows.append({"model": tag, "scheme": scheme, "epsilon": e,
                         "attack_to_benign_fnr": rc["attack_to_benign_fnr"][i],
                         "evasion_rate": rc["evasion_rate"][i]})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(TAB, "hardening_robustness.csv"), index=False)

piv = df[df.scheme == "pgd_grad"].pivot(index="epsilon", columns="model", values="evasion_rate")
print("\n=== PGD evasion_rate: plain vs hardened ===\n", piv.round(3).to_string())
e01 = df[(df.scheme == "pgd_grad") & (df.epsilon == 0.1)].set_index("model")["evasion_rate"]
print(f"\nPGD evasion @eps=0.1:  plain={e01['plain']:.3f} -> hardened={e01['hardened']:.3f}")

# summary table for the thesis
summ = pd.DataFrame([
    {"model": "plain CNN", "clean_macro_f1": round(rp["macro_f1"], 4),
     "FGSM_evasion@0.1": round(df[(df.model=='plain')&(df.scheme=='fgsm_grad')&(df.epsilon==0.1)]['evasion_rate'].iloc[0], 3),
     "PGD_evasion@0.1": round(e01["plain"], 3)},
    {"model": "hardened CNN", "clean_macro_f1": round(rh["macro_f1"], 4),
     "FGSM_evasion@0.1": round(df[(df.model=='hardened')&(df.scheme=='fgsm_grad')&(df.epsilon==0.1)]['evasion_rate'].iloc[0], 3),
     "PGD_evasion@0.1": round(e01["hardened"], 3)},
])
summ.to_csv(os.path.join(TAB, "hardening_summary.csv"), index=False)
print("\n=== hardening summary ===\n", summ.to_string(index=False))
print("\nsaved -> results/models/cnn_hardened.pt & results/tables/hardening_*.csv")
