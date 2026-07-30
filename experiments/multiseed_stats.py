# -*- coding: utf-8 -*-
"""Formal paired comparison of the repeated-run macro-F1 (supervisor: use a proper
test, do not infer significance from non-overlapping CIs alone).

Models are trained on the SAME five seeds, so seed-matched pairing is appropriate. For
each model pair we report the mean paired difference, its 95% t-CI, a paired t-test and
a Wilcoxon signed-rank test. With n = 5 the Wilcoxon two-sided p cannot fall below
0.0625, so results are described cautiously as observed differences."""
import os, sys, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy import stats

TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
d = pd.read_csv(os.path.join(TAB, "multiseed_ci_rows.csv"))
piv = d.pivot(index="seed", columns="model", values="macro_f1").sort_index()
models = list(piv.columns)
print("per-seed macro-F1:\n", piv.round(4).to_string(), "\n")

pairs = [("RandomForest", "DNN"), ("RandomForest", "MLP (sklearn)"), ("RandomForest", "1D-CNN"),
         ("XGBoost", "DNN"), ("XGBoost", "MLP (sklearn)"), ("XGBoost", "1D-CNN"),
         ("MLP (sklearn)", "DNN")]
rows = []
for a, b in pairs:
    if a not in models or b not in models:
        continue
    da, db = piv[a].values, piv[b].values
    diff = da - db; n = len(diff)
    md = diff.mean(); sd = diff.std(ddof=1)
    hw = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    t_p = stats.ttest_rel(da, db).pvalue
    try:
        w_p = stats.wilcoxon(da, db).pvalue
    except ValueError:
        w_p = float("nan")
    rows.append({"model_A": a, "model_B": b, "mean_diff": round(md, 4),
                 "diff_ci95_low": round(md - hw, 4), "diff_ci95_high": round(md + hw, 4),
                 "paired_t_p": round(float(t_p), 4), "wilcoxon_p": round(float(w_p), 4),
                 "A_higher_all_seeds": bool((diff > 0).all())})
    print(f"{a:14s} vs {b:14s} Δ={md:+.4f} [{md-hw:+.4f},{md+hw:+.4f}] "
          f"t-p={t_p:.4f} W-p={w_p:.4f} A>B all seeds={bool((diff>0).all())}")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(TAB, "multiseed_stats.csv"), index=False)
print("\nsaved -> results/tables/multiseed_stats.csv")
print("Note: n=5 (Wilcoxon two-sided p floor = 0.0625). Differences are reported as")
print("observed performance/stability; formal significance is claimed only where the")
print("paired t-test p < 0.05, and even then with the small-sample caveat.")
