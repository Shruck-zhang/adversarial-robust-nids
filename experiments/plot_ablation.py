# -*- coding: utf-8 -*-
"""Imbalance-strategy ablation figure: overall metrics + the rare-class
recall/precision trade-off (zoomed, because the differences are small)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

BASE = os.path.dirname(__file__)
TAB = os.path.join(BASE, "..", "results", "tables")
ov = pd.read_csv(os.path.join(TAB, "imbalance_ablation_overall.csv"), index_col=0)
rare = pd.read_csv(os.path.join(TAB, "imbalance_ablation_rare.csv"), index_col=0)

strat = list(ov.index)
x = np.arange(len(strat))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

# panel 1: overall
w = 0.38
a1.bar(x - w/2, ov["macro_f1"], w, label="macro-F1", color="#3274A1")
a1.bar(x + w/2, ov["balanced_accuracy"], w, label="balanced acc.", color="#E1812C")
a1.set_ylim(0.96, 0.99); a1.set_xticks(x); a1.set_xticklabels(strat, rotation=15, ha="right")
a1.set_title("Overall (zoomed)"); a1.legend(frameon=False); a1.grid(axis="y", alpha=0.3)
for i in x:
    a1.text(i - w/2, ov["macro_f1"].iloc[i], f"{ov['macro_f1'].iloc[i]:.4f}", ha="center", va="bottom", fontsize=7)

# panel 2: WebAttack recall vs precision (the only rare class that moves)
a2.bar(x - w/2, rare["WebAttack_recall"], w, label="recall", color="#3A923A")
a2.bar(x + w/2, rare["WebAttack_prec"], w, label="precision", color="#9467BD")
a2.set_ylim(0.94, 1.0); a2.set_xticks(x); a2.set_xticklabels(strat, rotation=15, ha="right")
a2.set_title("WebAttack: recall vs precision"); a2.legend(frameon=False); a2.grid(axis="y", alpha=0.3)

fig.suptitle("Imbalance-strategy ablation (XGBoost, same test set)")
fig.tight_layout()
out = os.path.join(BASE, "..", "results", "figures", "imbalance_ablation.png")
fig.savefig(out, dpi=150)
print("saved", out)
