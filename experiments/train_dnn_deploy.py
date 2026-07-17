# -*- coding: utf-8 -*-
"""Train the deployable hardened DNN on the FULL training data and save it.
Pairs with the existing full-data XGBoost for the two-tier system."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, joblib
import models_extra as MX
import metrics as MET

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
L = lambda n: np.load(os.path.join(PROC, n + ".npy"), allow_pickle=True)
Xtr, ytr = L("X_train"), L("y_train")
Xte, yte = L("X_test"), L("y_test")
cn = list(joblib.load(os.path.join(PROC, "class_names.pkl")))
nf, nc = Xtr.shape[1], len(cn)

# capped val for fast early-stopping
rng = np.random.RandomState(42); parts = []
yva = L("y_val"); Xva = L("X_val")
for c in range(nc):
    idx = np.where(yva == c)[0]
    if len(idx) > 5000:
        idx = rng.choice(idx, 5000, replace=False)
    parts.append(idx)
sel = np.concatenate(parts); Xvs, yvs = Xva[sel], yva[sel]
print(f"train {Xtr.shape}  val {Xvs.shape}  test {Xte.shape}", flush=True)

t = time.time()
hard = MX.train_net_adversarial(MX.build_dnn_net(nf, nc), Xtr, ytr, Xvs, yvs, nc,
                                eps=0.1, weight_scheme="sqrt", max_epochs=18,
                                patience=6, verbose=True, tag="dnn-adv")
print(f"hardened DNN trained in {time.time()-t:.0f}s", flush=True)
MX.save_dnn(hard, os.path.join(MODELS, "dnn_hardened.pt"), nc)

o = MET.evaluate(yte, hard.predict(Xte), cn)["overall"]
print(f"\nhardened DNN test: macroF1={o['macro_f1']:.4f}  balAcc={o['balanced_accuracy']:.4f}  "
      f"MCC={o['mcc']:.4f}  macroFPR={o['macro_fpr']:.4f}")
print("saved -> results/models/dnn_hardened.pt")
