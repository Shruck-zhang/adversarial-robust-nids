# -*- coding: utf-8 -*-
"""O4 (in-dataset) — attempted-attack detectability analysis.

The corrected CICIDS2017 uniquely labels payload-less "attempted" attacks. We
merged them into their parent class but kept an is_attempted flag; here we ask a
question only this dataset enables: do detectors catch attacks that were
*initiated but carried no payload* as well as effective ones? For each attack
class with enough attempted flows we compare the detection rate (flagged as any
attack) and class recall on effective vs attempted test flows, across models.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib
import models as M

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xte, yte, att = L("X_test"), L("y_test"), L("att_test").astype(bool)
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
BEN = cn.index("Benign")

classes = [c for c in range(len(cn)) if (att & (yte == c)).sum() >= 20]
print("classes with >=20 attempted flows:", [cn[c] for c in classes])

def load(name):
    if name == "1D-CNN":
        return M.load_cnn(os.path.join(MODELS, "cnn.pt"))
    return joblib.load(os.path.join(MODELS, name + ".pkl"))

rows = []
for name in ["XGBoost", "RandomForest", "MLP", "1D-CNN"]:
    try:
        mdl = load(name); pred = mdl.predict(Xte)
    except Exception as e:
        print(f"[skip] {name}: {e}"); continue
    for c in classes:
        eff = (yte == c) & ~att; at = (yte == c) & att
        rows.append({
            "model": name, "class": cn[c],
            "n_effective": int(eff.sum()), "n_attempted": int(at.sum()),
            "detect_effective": round(float((pred[eff] != BEN).mean()), 3),
            "detect_attempted": round(float((pred[at] != BEN).mean()), 3),
            "recall_effective": round(float((pred[eff] == c).mean()), 3),
            "recall_attempted": round(float((pred[at] == c).mean()), 3),
        })
    print(f"[done] {name}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(TAB, "attempted_analysis.csv"), index=False)
print("\n=== detection rate (flagged as any attack): effective vs attempted ===")
print(df.pivot_table(index="class", columns="model",
                     values=["detect_effective", "detect_attempted"]).round(3).to_string())
print("\nfull table saved -> results/tables/attempted_analysis.csv")
print(df.to_string(index=False))
