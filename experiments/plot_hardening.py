# -*- coding: utf-8 -*-
"""O3 figure: adversarial-training robustness gain (plain vs hardened CNN)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "..", "results", "tables", "hardening_robustness.csv"))

fig, ax = plt.subplots(figsize=(7.6, 4.6))
styles = {
    ("plain", "pgd_grad"):    ("#C62828", "-",  "plain — PGD"),
    ("plain", "fgsm_grad"):   ("#E1812C", "--", "plain — FGSM"),
    ("hardened", "pgd_grad"): ("#2E7D32", "-",  "hardened — PGD"),
    ("hardened", "fgsm_grad"):("#66BB6A", "--", "hardened — FGSM"),
}
for (m, s), (c, ls, lbl) in styles.items():
    d = df[(df.model == m) & (df.scheme == s)]
    ax.plot(d["epsilon"], d["evasion_rate"], ls, color=c, marker="o", ms=4, label=lbl)
ax.set_xlabel("perturbation strength  ε  (fraction of feature IQR)")
ax.set_ylabel("evasion rate")
ax.set_title("Adversarial training: robustness before vs after (O3)")
ax.grid(alpha=0.3); ax.legend(frameon=False)
ax.annotate("PGD@0.1: 0.71 → 0.05", xy=(0.1, 0.71), xytext=(0.12, 0.55),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9)
fig.tight_layout()
out = os.path.join(BASE, "..", "results", "figures", "hardening.png")
fig.savefig(out, dpi=150)
print("saved", out)
