# -*- coding: utf-8 -*-
"""Regenerate the (identical) leakage-free split but also keep the pre-scale
physical feature values, so the rule layer can be built and validated on raw
CICFlowMeter units. Also prints benign-vs-attack distributions of the candidate
rule features to guide high-precision threshold selection."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, joblib
import data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
X, y, att = D.load_clean_dataset(attempted_policy="merge", verbose=True)
sp = D.make_splits(X, y, is_att=att, random_state=42, verbose=True, return_raw=True)

# sanity: labels must line up with the already-saved arrays
y_test_saved = np.load(os.path.join(OUT, "y_test.npy"))
assert np.array_equal(sp["y_test"], y_test_saved), "split mismatch!"
np.save(os.path.join(OUT, "X_test_raw.npy"), sp["X_test_raw"])
np.save(os.path.join(OUT, "X_val_raw.npy"), sp["X_val_raw"])
print("saved X_test_raw / X_val_raw ; aligned with existing labels OK")

fn = sp["feature_names"]; cn = sp["class_names"]
Xr, yr = sp["X_val_raw"], sp["y_val"]
idx = {f: i for i, f in enumerate(fn)}
keyfeat = ["Dst Port", "Protocol", "Flow Duration", "Total Fwd Packet",
           "Total Length of Fwd Packet", "Bwd Packet Length Max",
           "Flow Packets/s", "Fwd Packets/s", "SYN Flag Count", "RST Flag Count",
           "Down/Up Ratio", "Fwd Act Data Pkts"]
print("\n=== per-class distributions on val_raw (median | p95) ===")
for f in keyfeat:
    j = idx[f]; print(f"\n[{f}]")
    for c, name in enumerate(cn):
        v = Xr[yr == c, j]
        if len(v) == 0:
            continue
        print(f"  {name:12s} n={len(v):>7d}  p50={np.median(v):>12.3f}  "
              f"p95={np.percentile(v,95):>12.3f}  max={v.max():>12.1f}")
