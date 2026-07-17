# -*- coding: utf-8 -*-
"""Stage-1 preprocessing: load+clean, leakage-free split + scale + prune,
cache arrays to data/processed/ for the Stage-2 baselines."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, joblib
import data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
X, y, att = D.load_clean_dataset(attempted_policy="merge", verbose=True)
sp = D.make_splits(X, y, is_att=att, random_state=42, verbose=True)
for k in ["X_train","y_train","att_train","X_val","y_val","att_val","X_test","y_test","att_test"]:
    np.save(os.path.join(OUT, k + ".npy"), sp[k])
joblib.dump(sp["scaler"], os.path.join(OUT, "scaler.pkl"))
joblib.dump(sp["feature_names"], os.path.join(OUT, "feature_names.pkl"))
joblib.dump(sp["class_names"], os.path.join(OUT, "class_names.pkl"))

import collections
def dist(yv):
    c = collections.Counter(yv.tolist()); return {sp['class_names'][i]: c.get(i,0) for i in range(len(sp['class_names']))}
print("\ntrain class counts:", dist(sp["y_train"]))
print("val   class counts:", dist(sp["y_val"]))
print("test  class counts:", dist(sp["y_test"]))
print(f"\nDONE in {time.time()-t0:.0f}s -> data/processed/ ({len(sp['feature_names'])} features)")
