# -*- coding: utf-8 -*-
"""Figures for the rigour analyses: repeated-run CIs, rare-class
successful-vs-attempted detection, and adaptive-vs-transfer ensemble evasion."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

BASE = os.path.dirname(__file__)
TAB = os.path.join(BASE, "..", "results", "tables")
FIG = os.path.join(BASE, "..", "results", "figures")

# 1 — repeated-run CIs
p = os.path.join(TAB, "multiseed_ci.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(d))
    ax.errorbar(x, d["macro_f1_mean"], yerr=d["ci95_halfwidth"], fmt="o", color="#1565C0",
                capsize=6, markersize=8, lw=2)
    ax.set_xticks(x); ax.set_xticklabels(d["model"], rotation=20, ha="right")
    ax.set_ylabel("macro-F1 (mean ± 95% CI, 5 seeds)")
    ax.set_title("Repeated-run confidence intervals (capped-budget comparison)")
    for xi, r in zip(x, d.itertuples()):
        ax.annotate(f"±{r.ci95_halfwidth:.3f}", (xi, r.macro_f1_mean), textcoords="offset points",
                    xytext=(9, 0), fontsize=8, color="#555", va="center")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "multiseed_ci.png"), dpi=150); plt.close(fig)
    print("saved multiseed_ci.png")

# 2 — rare class: successful vs attempted detection
p = os.path.join(TAB, "rare_class_attempted.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    d = d[d["successful_n"].astype(str) != ""]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(d)); w = 0.38
    succ = pd.to_numeric(d["successful_detect"], errors="coerce").fillna(0)
    att = pd.to_numeric(d["attempted_detect"], errors="coerce").fillna(0)
    ax.bar(x - w/2, succ, w, label="successful", color="#2E7D32")
    ax.bar(x + w/2, att, w, label="attempted", color="#C62828")
    ax.set_xticks(x); ax.set_xticklabels(d["class"], rotation=20, ha="right")
    ax.set_ylabel("detection rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-class detection: successful vs attempted attacks")
    for xi, a, n in zip(x, att, d["attempted_n"]):
        if n and not np.isnan(a):
            ax.annotate(f"{a:.2f}", (xi + w/2, a), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color="#C62828")
    ax.legend(frameon=False); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rare_class_attempted.png"), dpi=150); plt.close(fig)
    print("saved rare_class_attempted.png")

# 3 — adaptive vs transfer two-tier evasion
p = os.path.join(TAB, "adaptive_ensemble_attack.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(d["epsilon"], d["transfer_two_tier"], "o--", color="#1565C0", label="transfer attack")
    ax.plot(d["epsilon"], d["adaptive_two_tier"], "D-", color="#C62828", lw=2, label="adaptive attack (50-step PGD, 5 restarts)")
    ax.set_xlabel("perturbation strength ε"); ax.set_ylabel("two-tier evasion (both tiers fooled)")
    ax.set_title("Two-tier evasion: transfer vs adaptive white-box attack")
    ax.set_ylim(0, max(0.1, d["adaptive_two_tier"].max() * 1.5))
    ax.grid(alpha=0.3); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "adaptive_ensemble.png"), dpi=150); plt.close(fig)
    print("saved adaptive_ensemble.png")
