# -*- coding: utf-8 -*-
"""Figure: detection rate on effective vs attempted attack flows, per class/model."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "..", "results", "tables", "attempted_analysis.csv"))
classes = ["DoS", "WebAttack", "Bot"]
models = ["XGBoost", "RandomForest", "MLP", "1D-CNN"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
x = np.arange(len(models)); w = 0.38
for ax, cls in zip(axes, classes):
    d = df[df["class"] == cls].set_index("model")
    eff = [d.loc[m, "detect_effective"] for m in models]
    att = [d.loc[m, "detect_attempted"] for m in models]
    ax.bar(x - w/2, eff, w, label="effective", color="#3274A1")
    ax.bar(x + w/2, att, w, label="attempted (no payload)", color="#E1812C")
    na = int(d["n_attempted"].iloc[0]); ne = int(d["n_effective"].iloc[0])
    ax.set_title(f"{cls}\n(eff={ne:,}, att={na:,})", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("detection rate (flagged as attack)")
axes[0].legend(loc="lower left", frameon=False, fontsize=8)
fig.suptitle("Detectability of effective vs 'attempted' (payload-less) attacks (O4)")
fig.tight_layout()
out = os.path.join(BASE, "..", "results", "figures", "attempted_analysis.png")
fig.savefig(out, dpi=150); print("saved", out)
