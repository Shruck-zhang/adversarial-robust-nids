# -*- coding: utf-8 -*-
"""Adaptive attack against the complete two-tier ensemble (supervisor point 3).

The transfer experiment crafts a weak PGD on the DNN and reads off the effect on
XGBoost. Here we test an *adaptive* white-box attacker that optimises harder against
the ensemble's binding constraint. Because XGBoost transfers easily (~0.63 evasion at
eps=0.1) while the adversarially-hardened DNN is the robust component, the adaptive
attacker concentrates on the DNN with a much stronger PGD (more steps, smaller step,
multiple random restarts) and we then require BOTH tiers to be evaded.

We report, at each epsilon, the two-tier evasion (both tiers -> benign) under the
original transfer attack vs the adaptive attack, so the robustness claim is either
strengthened (stays low) or honestly qualified (rises)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib
import torch, torch.nn.functional as F
import security as S

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import two_tier
Xte = np.load(os.path.join(PROC, "X_test.npy")).astype("float32")
yte = np.load(os.path.join(PROC, "y_test.npy"))
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl"))); BEN = cn.index("Benign")
det = two_tier.TwoTierDetector(MODELS, PROC)
dnn, xgb, net = det.dnn, det.xgb, det.dnn.net; net.eval()

rng = np.random.RandomState(0)
sel = np.concatenate([rng.choice(np.where(yte == c)[0], min(300, int((yte == c).sum())), replace=False)
                      for c in range(len(cn)) if c != BEN])
X0, y0 = Xte[sel], yte[sel]
print(f"attack flows: {len(X0)}", flush=True)

def pgd_from(start, X0n, yn, eps, steps, alpha):
    x = torch.tensor(start, dtype=torch.float32)
    Xb = torch.tensor(X0n, dtype=torch.float32); yb = torch.tensor(yn, dtype=torch.long)
    for _ in range(steps):
        x.requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(net(x), yb), x)[0]
        x = (x.detach() + alpha * g.sign())
        x = torch.max(torch.min(x, Xb + eps), Xb - eps).clamp(-10, 10)
    return x.detach().numpy().astype("float32")

def true_conf(adv, yn):
    """DNN softmax probability of the true class (lower = more evasive)."""
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor(adv, dtype=torch.float32)), 1).numpy()
    return p[np.arange(len(yn)), yn]

def adaptive(X0n, yn, eps, steps=50, restarts=5):
    """Multi-restart strong PGD; keep, per sample, the MOST evasive adversarial example
    (lowest true-class confidence). Always returns a real adv (never the clean sample),
    so every downstream tier statistic is measured on genuine perturbations."""
    alpha = eps / 10.0
    best, bestc = None, None
    for r in range(restarts):
        start = X0n if r == 0 else np.clip(X0n + rng.uniform(-eps, eps, X0n.shape), -10, 10).astype("float32")
        adv = pgd_from(start, X0n, yn, eps, steps, alpha)
        c = true_conf(adv, yn)
        if best is None:
            best, bestc = adv.copy(), c
        else:
            m = c < bestc
            best[m] = adv[m]; bestc[m] = c[m]
    return best

rows = []
for eps in [0.05, 0.1, 0.2]:
    # transfer baseline = the original 10-step PGD on the DNN
    Xtr = S.pgd_grad(dnn, X0, y0, eps)
    t_dnn = float((dnn.predict(Xtr) == BEN).mean())
    t_xgb = float((xgb.predict(Xtr) == BEN).mean())
    t_two = float(((dnn.predict(Xtr) == BEN) & (xgb.predict(Xtr) == BEN)).mean())
    # adaptive = strong multi-restart PGD on the DNN
    t = time.time(); Xad = adaptive(X0, y0, eps)
    a_dnn = float((dnn.predict(Xad) == BEN).mean())
    a_xgb = float((xgb.predict(Xad) == BEN).mean())
    a_two = float(((dnn.predict(Xad) == BEN) & (xgb.predict(Xad) == BEN)).mean())
    rows.append({"epsilon": eps,
                 "transfer_dnn": round(t_dnn, 3), "transfer_two_tier": round(t_two, 3),
                 "adaptive_dnn": round(a_dnn, 3), "adaptive_xgb": round(a_xgb, 3),
                 "adaptive_two_tier": round(a_two, 3)})
    print(f"eps={eps}: transfer two-tier={t_two:.3f} | adaptive DNN={a_dnn:.3f} "
          f"XGB={a_xgb:.3f} two-tier={a_two:.3f}  ({time.time()-t:.0f}s)", flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(TAB, "adaptive_ensemble_attack.csv"), index=False)
print("\n" + out.to_string(index=False))
print("\nsaved -> results/tables/adaptive_ensemble_attack.csv")
print("interpretation: adaptive two-tier evasion vs transfer two-tier evasion at each eps")
print("shows whether a stronger attacker on the ensemble's robust component (the hardened")
print("DNN) meaningfully raises full-system evasion. Attack is white-box, unconstrained in")
print("feature space (conservative upper bound; not necessarily feasible traffic).")
