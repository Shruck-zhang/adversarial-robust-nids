# -*- coding: utf-8 -*-
"""Two-tier defence-in-depth figure: evasion vs epsilon for the DNN alone, the
XGBoost tier on the same (DNN-crafted) perturbation, and the two-tier system."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "..", "results", "tables", "two_tier_robustness.csv"))

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(df["epsilon"], df["evasion_DNN_only"], "o-", color="#C62828", label="hardened DNN alone")
ax.plot(df["epsilon"], df["evasion_XGB_on_adv"], "s--", color="#1565C0", label="XGBoost (on DNN-crafted adv.)")
ax.plot(df["epsilon"], df["evasion_two_tier"], "D-", color="#2E7D32", lw=2.5, label="two-tier (evades BOTH)")
ax.set_xlabel("perturbation strength  ε  (PGD crafted on the DNN)")
ax.set_ylabel("attack evasion rate")
ax.set_title("Two-tier defence in depth: an attacker must fool both tiers")
ax.grid(alpha=0.3); ax.legend(frameon=False)
fig.tight_layout()
out = os.path.join(BASE, "..", "results", "figures", "two_tier_robustness.png")
fig.savefig(out, dpi=150); print("saved", out)
