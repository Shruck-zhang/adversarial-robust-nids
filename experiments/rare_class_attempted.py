# -*- coding: utf-8 -*-
"""Rare-class analysis split by successful vs attempted attacks.

The corrected CICIDS2017 tags payload-less *attempted* attacks with a flag (stored
as metadata, never a model feature). Here the deployed detector's per-class detection
rate is reported SEPARATELY for successful and attempted attacks, exposing the
attempted-attack blind spot (e.g. attempted DoS)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib
from detection_engine import DetectionEngine

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
Xraw = np.load(os.path.join(PROC, "X_test_raw.npy"))
y = np.load(os.path.join(PROC, "y_test.npy"))
att = np.load(os.path.join(PROC, "att_test.npy")).astype(bool)
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
eng = DetectionEngine(MODELS, PROC); BEN = eng.benign
det = eng.predict_detailed(Xraw)
is_atk = det["is_attack"]

rows = []
print(f"{'class':13s}{'succ n':>8s}{'succ det':>10s}{'att n':>8s}{'att det':>10s}")
print("-" * 52)
for c, name in enumerate(cn):
    if c == BEN:
        continue
    m_s = (y == c) & ~att
    m_a = (y == c) & att
    ns, na = int(m_s.sum()), int(m_a.sum())
    ds = float(is_atk[m_s].mean()) if ns else float("nan")
    da = float(is_atk[m_a].mean()) if na else float("nan")
    rows.append({"class": name, "successful_n": ns, "successful_detect": round(ds, 4) if ns else "",
                 "attempted_n": na, "attempted_detect": round(da, 4) if na else ""})
    print(f"{name:13s}{ns:>8d}{ds:>10.4f}{na:>8d}{(da if na else float('nan')):>10.4f}")

pd.DataFrame(rows).to_csv(os.path.join(TAB, "rare_class_attempted.csv"), index=False)
print("\nsaved -> results/tables/rare_class_attempted.csv")
print("note: 'attempted' = payload-less attack variants; low attempted-detection is a")
print("known blind spot and is reported separately so it does not inflate headline recall.")
