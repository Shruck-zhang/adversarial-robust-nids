# -*- coding: utf-8 -*-
"""Stage-2 figure: per-class recall (detection rate) across baseline models,
with the rare classes highlighted."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

BASE = os.path.dirname(__file__)
TAB = os.path.join(BASE, "..", "results", "tables")
rec = pd.read_csv(os.path.join(TAB, "baselines_per_class_recall.csv"), index_col=0)

order = ["Benign", "DoS", "DDoS", "PortScan", "Infiltration", "BruteForce", "WebAttack", "Bot"]
rec = rec.reindex([c for c in order if c in rec.index])
rare = {"BruteForce", "WebAttack", "Bot"}

models = list(rec.columns)
x = np.arange(len(rec)); w = 0.8 / len(models)
colors = ["#3274A1", "#E1812C", "#3A923A", "#9467BD"]
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, m in enumerate(models):
    ax.bar(x + i * w, rec[m].values, w, label=m, color=colors[i % len(colors)])
ax.set_xticks(x + w * (len(models) - 1) / 2)
ax.set_xticklabels([(c + "  ★" if c in rare else c) for c in rec.index], rotation=25, ha="right")
ax.set_ylabel("Recall (detection rate)")
ax.set_ylim(0, 1.02)
ax.set_title("Per-class recall by baseline model  (★ = rare class)")
ax.legend(loc="lower left", frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = os.path.join(BASE, "..", "results", "figures", "baselines_per_class_recall.png")
fig.savefig(out, dpi=150)
print("saved", out)
