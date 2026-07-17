# -*- coding: utf-8 -*-
"""Wire the whole system together and debug the interfaces end-to-end.

Stage A  real CICFlowMeter output   -> adapter -> engine   (interface / schema check)
Stage B  labelled replay            -> engine              (detection correctness + throughput)
Stage C  streaming live-simulation  -> run() loop          (alerts + rolling status)
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib
import metrics as MET
from detection_engine import DetectionEngine
import realtime as RT

BASE = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "results", "models")
SAMPLE = os.path.join(BASE, "data", "sample_flows", "sample_cicflowmeter_Flow.csv")

eng = DetectionEngine(MODELS, PROC)
ids = RT.RealTimeIDS(eng)
cn = eng.class_names; BEN = eng.benign

# ============ Stage A: real CICFlowMeter CSV through the adapter ============ #
print("=" * 74, "\nSTAGE A — real CICFlowMeter output -> adapter -> engine\n" + "=" * 74)
df = pd.read_csv(SAMPLE)
print(f"loaded {SAMPLE.split(os.sep)[-1]}: {len(df):,} rows, {df.shape[1]} columns")
valid = RT.FlowAdapter.valid_mask(df)
print(f"valid flows (duration>0): {int(valid.sum()):,}")
X, meta, missing = ids.adapter.adapt(df[valid].reset_index(drop=True))
print(f"adapted matrix: {X.shape}  (expected 66 cols)")
print(f"features not supplied by this CICFlowMeter build -> zero-filled ({len(missing)}): {missing}")
res = ids.score_matrix(X, meta)
print(f"scored {res['n']:,} real flows -> {len(res['alerts']):,} attack alerts")
print("sample verdicts (first 6 attack alerts on real traffic):")
for a in res["alerts"][:6]:
    print("   " + a.line())
print("interface OK — real CICFlowMeter rows flow through end-to-end.\n")

# ============ Stage B: labelled replay — correctness + throughput ============ #
print("=" * 74, "\nSTAGE B — labelled replay (ground truth) -> detection correctness\n" + "=" * 74)
Xraw = np.load(os.path.join(PROC, "X_test_raw.npy"))
y = np.load(os.path.join(PROC, "y_test.npy"))
rng = np.random.RandomState(0)
parts = [rng.choice(np.where(y == c)[0], min((y == c).sum(), 8000), replace=False)
         for c in range(len(cn))]
sel = np.concatenate(parts); Xs, ys = Xraw[sel], y[sel]
ids.reset_stats()
t = time.time()
det = eng.predict_detailed(Xs)
dt = time.time() - t
pred = det["final"]
o = MET.evaluate(ys, pred, cn)["overall"]
det_rate = float((det["is_attack"][ys != BEN]).mean())
fa = float((det["is_attack"][ys == BEN]).mean())
print(f"replayed {len(Xs):,} labelled flows")
print(f"  attack detection rate (recall on attacks): {det_rate:.3f}")
print(f"  benign false-alarm rate:                   {fa:.3f}")
print(f"  macro-F1 / balanced-acc:                   {o['macro_f1']:.3f} / {o['balanced_accuracy']:.3f}")
print(f"  throughput: {len(Xs)/dt:,.0f} flows/s  ({1e6*dt/len(Xs):.1f} us/flow)  -> real-time capable\n")

# ============ Stage C: streaming live simulation ============ #
print("=" * 74, "\nSTAGE C — streaming simulation (benign background + injected attacks)\n" + "=" * 74)
# a realistic window: mostly benign with a burst of mixed attacks
ben = rng.choice(np.where(y == BEN)[0], 4000, replace=False)
atk = np.concatenate([rng.choice(np.where(y == c)[0], 120, replace=False)
                      for c in range(len(cn)) if c != BEN])
stream_idx = rng.permutation(np.concatenate([ben, atk]))
Xstream = Xraw[stream_idx]
ids.reset_stats()
shown = [0]
def on_alert(a):
    if shown[0] < 8:
        print("  " + a.line()); shown[0] += 1
ids.run(RT.iter_replay_df(Xstream, eng.feature_names, batch=512),
        on_alert=on_alert, status_every=2000)
print("\nall interfaces connected and debugged: CSV/adapter, engine, streaming runner.")
