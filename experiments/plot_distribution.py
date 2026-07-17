# -*- coding: utf-8 -*-
"""Stage-1 figure: 8-class distribution (log scale) with attempted overlay."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "..", "results", "tables", "class_distribution.csv"), index_col=0)
df = df.sort_values("count", ascending=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
eff = df["count"] - df["attempted"]
ax.barh(df.index, eff, color="#3274A1", label="effective")
ax.barh(df.index, df["attempted"], left=eff, color="#E1812C", label="attempted")
ax.set_xscale("log")
ax.set_xlabel("Number of flows (log scale)")
ax.set_title("Improved CICIDS2017 — 8-class distribution (train pool, after cleaning)")
for i, (c, r) in enumerate(df.iterrows()):
    ax.text(r["count"] * 1.05, i, f"{int(r['count']):,} ({r['pct']:.2f}%)",
            va="center", fontsize=8, color="#333")
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(right=df["count"].max() * 3)
fig.tight_layout()
out = os.path.join(BASE, "..", "results", "figures", "class_distribution.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("saved", out)
