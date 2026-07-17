# -*- coding: utf-8 -*-
"""Small human-readable demo of the two-layer detection engine: pick one real flow
of each class from the raw test set and print the engine's verdict + explanation."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, joblib
from detection_engine import DetectionEngine

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
Xraw = np.load(os.path.join(PROC, "X_test_raw.npy"))
y = np.load(os.path.join(PROC, "y_test.npy"))
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
eng = DetectionEngine(MODELS, PROC)

# one representative flow per true class
rng = np.random.RandomState(7)
pick = [int(rng.choice(np.where(y == c)[0])) for c in range(len(cn))]
sub = Xraw[pick]
d = eng.predict_detailed(sub)
print("=== two-layer detection engine — sample verdicts ===\n")
print(f"{'true':13s}{'-> verdict':13s}{'via':7s} reason")
print("-" * 92)
for k in range(len(pick)):
    true = cn[y[pick[k]]]
    verdict = cn[int(d['final'][k])]
    via = d['source'][k]
    reason = d['reason'][k] or ("AI anomaly (XGBoost/DNN)" if via in ("ai", "both") else "-")
    flag = "  [!]" if d['disagree'][k] else ""
    print(f"{true:13s}-> {verdict:11s}{via:7s} {reason}{flag}")
