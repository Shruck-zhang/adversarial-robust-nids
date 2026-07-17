# -*- coding: utf-8 -*-
"""Comprehensive model comparison figure — macro-F1 ranked, coloured by family."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "..", "results", "tables", "model_comparison.csv"))
df = df.sort_values("macro_f1")

FAM = {"RandomForest":"tree","XGBoost":"tree","DecisionTree":"tree",
       "LogisticRegression":"linear","MLP":"deep-FF","DNN":"deep-FF",
       "1D-CNN":"CNN","LSTM":"recurrent","GRU":"recurrent",
       "CNN-LSTM":"hybrid","TabNet":"attention"}
COL = {"tree":"#3274A1","linear":"#8C8C8C","deep-FF":"#E1812C","CNN":"#C62828",
       "recurrent":"#6A1B9A","hybrid":"#2E7D32","attention":"#00838F"}
colors = [COL[FAM[m]] for m in df["model"]]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(df["model"], df["macro_f1"], color=colors)
for i, (m, v) in enumerate(zip(df["model"], df["macro_f1"])):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlim(0.65, 1.0); ax.set_xlabel("macro-F1 (test set)")
ax.set_title("Comprehensive model comparison on corrected CICIDS2017 (O1)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=f) for f, c in COL.items()],
          loc="lower right", frameon=False, fontsize=8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
out = os.path.join(BASE, "..", "results", "figures", "model_comparison.png")
fig.savefig(out, dpi=150)
print("saved", out)
