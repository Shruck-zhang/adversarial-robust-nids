# -*- coding: utf-8 -*-
"""O2-O3 across architectures — multi-model adversarial robustness comparison.

For each differentiable architecture (MLP, DNN, 1D-CNN) it trains a plain and an
adversarially-hardened version, then measures clean macro-F1 and the FGSM/PGD
evasion rate — answering "which architecture is most robust, and how much does
adversarial training help each?". Resumable (per model) and CPU-tractable
(capped train + capped attack-eval sample).
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib

import models as M
import models_extra as MX
import metrics as MET
import security as S

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
os.makedirs(TAB, exist_ok=True)
SUMM = os.path.join(TAB, "robustness_comparison.csv")
CURV = os.path.join(TAB, "robustness_comparison_curves.csv")
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xtr0, ytr0 = L("X_train"), L("y_train")
Xva, yva = L("X_val"), L("y_val")
Xte, yte = L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
nf, nc = Xtr0.shape[1], len(cn); BEN = cn.index("Benign")

def capset(X, y, per, benign_cap=None, seed=42):
    rng = np.random.RandomState(seed); keep = []
    for c in range(nc):
        idx = np.where(y == c)[0]
        cc = benign_cap if (benign_cap and c == BEN) else per
        if len(idx) > cc:
            idx = rng.choice(idx, cc, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep); rng.shuffle(keep)
    return X[keep], y[keep]

Xtr, ytr = capset(Xtr0, ytr0, 25000)
Xvs, yvs = capset(Xva, yva, 5000)
Xs, ys = capset(Xte, yte, 2000, benign_cap=15000)   # attack-eval sample
print(f"train {Xtr.shape}  val {Xvs.shape}  attack-eval {Xs.shape}", flush=True)
Xs = Xs.astype("float32")
EPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

NETS = {"MLP":    lambda: MX.build_mlp_net(nf, nc),
        "DNN":    lambda: MX.build_dnn_net(nf, nc),
        "1D-CNN": lambda: M._build_cnn1d_net(nf, nc)}

done = set(pd.read_csv(SUMM)["model"]) if os.path.exists(SUMM) else set()
print(f"already done: {sorted(done)}", flush=True)

def ev_at(rc, e):
    return rc["evasion_rate"][EPS.index(e)]

for name, netfn in NETS.items():
    if name in done:
        print(f"[skip] {name}", flush=True); continue
    t = time.time()
    plain = MX.train_net(netfn(), Xtr, ytr, Xvs, yvs, nc, tag=name)
    hard = MX.train_net_adversarial(netfn(), Xtr, ytr, Xvs, yvs, nc, tag=name + "-adv")
    print(f"  {name}: trained plain+hardened in {time.time()-t:.0f}s", flush=True)

    summ_rows, curve_rows = [], []
    for variant, mdl in [("plain", plain), ("hardened", hard)]:
        clean = MET.evaluate(yte, mdl.predict(Xte), cn)["overall"]["macro_f1"]
        rc_f = S.robustness_curve(mdl, Xs, ys, EPS, scheme="fgsm_grad", n_classes=nc, benign_index=BEN)
        rc_p = S.robustness_curve(mdl, Xs, ys, EPS, scheme="pgd_grad", n_classes=nc, benign_index=BEN)
        for scheme, rc in [("fgsm_grad", rc_f), ("pgd_grad", rc_p)]:
            for i, e in enumerate(EPS):
                curve_rows.append({"model": name, "variant": variant, "scheme": scheme,
                                   "epsilon": e, "evasion_rate": rc["evasion_rate"][i],
                                   "attack_to_benign_fnr": rc["attack_to_benign_fnr"][i]})
        summ_rows.append({"model": name, "variant": variant,
                          "clean_macro_f1": round(clean, 4),
                          "fgsm_evasion@0.1": round(ev_at(rc_f, 0.1), 3),
                          "pgd_evasion@0.1": round(ev_at(rc_p, 0.1), 3)})

    # append (resumable)
    for path, rows in [(SUMM, summ_rows), (CURV, curve_rows)]:
        new = pd.DataFrame(rows)
        df = pd.concat([pd.read_csv(path), new], ignore_index=True) if os.path.exists(path) else new
        df.to_csv(path, index=False)
    print(f"[done] {name}: " + " | ".join(
        f"{r['variant']} clean={r['clean_macro_f1']} PGD@0.1={r['pgd_evasion@0.1']}" for r in summ_rows), flush=True)

print("\n=== multi-model robustness (test) ===")
print(pd.read_csv(SUMM).to_string(index=False))
