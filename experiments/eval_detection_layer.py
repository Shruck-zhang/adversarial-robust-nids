# -*- coding: utf-8 -*-
"""Validate the two-layer detection engine on the raw test split.

Reports (1) the rule layer alone — per-rule precision and coverage, and its false
alarm rate on benign traffic; (2) the full engine vs the AI-only two-tier — clean
macro metrics, the fraction of attacks the rule layer catches on the fast path, and
rule/AI disagreement statistics."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib
import metrics as MET
from detection_engine import DetectionEngine

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
os.makedirs(TAB, exist_ok=True)

Xraw = np.load(os.path.join(PROC, "X_test_raw.npy"))
y = np.load(os.path.join(PROC, "y_test.npy"))
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
BEN = cn.index("Benign")
eng = DetectionEngine(MODELS, PROC)
print(f"test flows: {len(y):,}   attacks: {int((y!=BEN).sum()):,}")

# ---------- Layer 1 alone: precision / coverage per rule ---------- #
rl = eng.rules.predict_detailed(Xraw)
rp, rname = rl["pred"], rl["rule"]
is_atk = y != BEN
print("\n=== Layer 1 (rules) — precision & coverage ===")
rows = []
for name in ["portscan", "bruteforce", "ddos"]:
    m = rname == name
    if m.sum() == 0:
        print(f"  {name:11s}: never fired"); continue
    claimed = rp[m][0]                         # class this rule asserts
    prec_atk = float(is_atk[m].mean())         # fired flows that are truly attacks
    prec_cls = float((y[m] == claimed).mean()) # fired flows that are truly that class
    # coverage of that specific class
    cov = float(m[y == claimed].mean()) if (y == claimed).any() else 0.0
    rows.append({"rule": name, "class": cn[claimed], "fired": int(m.sum()),
                 "precision_attack": round(prec_atk, 4),
                 "precision_class": round(prec_cls, 4),
                 "recall_of_class": round(cov, 4)})
    print(f"  {name:11s} -> {cn[claimed]:10s} fired={m.sum():>7d}  "
          f"P(attack)={prec_atk:.3f}  P(class)={prec_cls:.3f}  R(class)={cov:.3f}")
any_rule = rp != BEN
false_alarm = float((any_rule & ~is_atk).sum()) / max(1, int((~is_atk).sum()))
print(f"\n  rule layer false-alarm rate on benign: {false_alarm:.5f} "
      f"({int((any_rule & ~is_atk).sum())} of {int((~is_atk).sum()):,} benign flows)")
pd.DataFrame(rows).to_csv(os.path.join(TAB, "detection_rules.csv"), index=False)

# ---------- Full engine vs AI-only ---------- #
det = eng.predict_detailed(Xraw)
final = det["final"]
ai_only = det["ai_pred"]
print("\n=== full engine vs AI-only (two-tier) ===")
comp = []
for label, pred in [("AI-only (two-tier)", ai_only), ("full engine (rules+AI)", final)]:
    o = MET.evaluate(y, pred, cn)["overall"]
    comp.append({"system": label, "macro_f1": round(o["macro_f1"], 4),
                 "balanced_acc": round(o["balanced_accuracy"], 4),
                 "macro_fpr": round(o["macro_fpr"], 4)})
compdf = pd.DataFrame(comp); print(compdf.to_string(index=False))
compdf.to_csv(os.path.join(TAB, "detection_engine.csv"), index=False)

# ---------- operational stats ---------- #
src = det["source"]
caught_atk = is_atk.sum()
by_rule = int(((src == "rule") & is_atk).sum())
by_both = int(((src == "both") & is_atk).sum())
fastpath = (by_rule + by_both) / max(1, int(caught_atk))
print(f"\nattacks explainable on the rule fast-path: {fastpath:.3f} "
      f"({by_rule+by_both:,} of {int(caught_atk):,})")
print(f"rule/AI disagreements flagged (possible manipulation): "
      f"{int(det['disagree'].sum()):,} flows")
print("\nsaved -> results/tables/detection_rules.csv, detection_engine.csv")
