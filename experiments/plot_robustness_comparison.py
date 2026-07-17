# -*- coding: utf-8 -*-
"""Multi-model robustness figure: PGD evasion and clean macro-F1, plain vs
hardened, for each differentiable architecture."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "..", "results", "tables", "robustness_comparison.csv"))
order = ["MLP", "DNN", "1D-CNN"]
df = df.set_index(["model", "variant"])
x = np.arange(len(order)); w = 0.38

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
# panel 1: PGD evasion @0.1 (lower = more robust)
a1.bar(x - w/2, [df.loc[(m, "plain"), "pgd_evasion@0.1"] for m in order], w,
       label="plain", color="#C62828")
a1.bar(x + w/2, [df.loc[(m, "hardened"), "pgd_evasion@0.1"] for m in order], w,
       label="hardened", color="#2E7D32")
a1.set_xticks(x); a1.set_xticklabels(order); a1.set_ylabel("PGD evasion @ ε=0.1")
a1.set_title("Robustness (lower = better)"); a1.legend(frameon=False); a1.grid(axis="y", alpha=0.3)
for i, m in enumerate(order):
    for off, v in [(-w/2, df.loc[(m,"plain"),"pgd_evasion@0.1"]), (w/2, df.loc[(m,"hardened"),"pgd_evasion@0.1"])]:
        a1.text(i+off, v+0.008, f"{v:.2f}", ha="center", fontsize=8)

# panel 2: clean macro-F1 (higher = better)
a2.bar(x - w/2, [df.loc[(m, "plain"), "clean_macro_f1"] for m in order], w,
       label="plain", color="#3274A1")
a2.bar(x + w/2, [df.loc[(m, "hardened"), "clean_macro_f1"] for m in order], w,
       label="hardened", color="#8AB6E0")
a2.set_xticks(x); a2.set_xticklabels(order); a2.set_ylabel("clean macro-F1")
a2.set_ylim(0.8, 1.0); a2.set_title("Clean accuracy (cost of hardening)")
a2.legend(frameon=False); a2.grid(axis="y", alpha=0.3)

fig.suptitle("Adversarial robustness across architectures (O2–O3)")
fig.tight_layout()
out = os.path.join(BASE, "..", "results", "figures", "robustness_comparison.png")
fig.savefig(out, dpi=150); print("saved", out)
