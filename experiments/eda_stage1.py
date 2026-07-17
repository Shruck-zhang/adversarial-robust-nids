# -*- coding: utf-8 -*-
"""Stage 1 EDA: load + clean the improved CICIDS2017, report the 8-class
distribution, imbalance, rare classes and the attempted-attack breakdown."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import data as D

RES = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
os.makedirs(RES, exist_ok=True)

t0 = time.time()
print("=== loading + cleaning (attempted_policy='merge') ===")
X, y, is_att = D.load_clean_dataset(attempted_policy="merge", verbose=True)
print(f"\nloaded in {time.time()-t0:.0f}s   X={X.shape}   features={X.shape[1]}")

# class distribution
vc = y.value_counts().reindex(D.CLASS_NAMES).fillna(0).astype(int)
att_by_cls = pd.Series(is_att.values, index=y.values).groupby(level=0).sum().reindex(D.CLASS_NAMES).fillna(0).astype(int)
dist = pd.DataFrame({
    "count": vc,
    "pct": (vc / vc.sum() * 100).round(3),
    "attempted": att_by_cls,
})
dist["attempted_pct_of_class"] = (dist["attempted"] / dist["count"].replace(0, np.nan) * 100).round(1)
print("\n=== 8-class distribution ===")
print(dist.to_string())

nz = vc[vc > 0]
print(f"\ntotal flows: {vc.sum():,}")
print(f"benign share: {vc.get('Benign',0)/vc.sum()*100:.2f}%")
print(f"imbalance ratio (max/min class): {nz.max()/nz.min():.0f} : 1")
print(f"rare classes (<1% share): {[c for c in nz.index if vc[c]/vc.sum()<0.01]}")
print(f"total attempted flows: {int(is_att.sum()):,} ({is_att.mean()*100:.2f}%)")

dist.to_csv(os.path.join(RES, "class_distribution.csv"))
X.dtypes.astype(str).to_csv(os.path.join(RES, "feature_schema.csv"), header=["dtype"])
print(f"\nsaved -> results/tables/class_distribution.csv  &  feature_schema.csv")
print(f"feature list ({X.shape[1]}): {list(X.columns)}")
