# -*- coding: utf-8 -*-
"""Deployment latency & throughput: model inference vs the full capture-to-alert pipeline.

Separates (a) model-inference from (b) the full engine (rules + scaling + two-tier)
and (c) the engine including the CICFlowMeter feature adapter, and reports hardware,
batch size and a latency DISTRIBUTION (percentiles), not just a mean. Also states the
capture-to-alert flow-completion floor that dominates real end-to-end latency."""
import os, sys, time, platform
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib
from detection_engine import DetectionEngine
import realtime as RT

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "results", "models")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
Xraw = np.load(os.path.join(PROC, "X_test_raw.npy")).astype("float64")
eng = DetectionEngine(MODELS, PROC)
fn = eng.feature_names

# ---- hardware ----
cores = os.cpu_count()
ram = None
try:
    import psutil; ram = round(psutil.virtual_memory().total / 1e9, 1)
except Exception:
    pass
hw = f"{platform.processor() or platform.machine()} | {cores} logical cores | {ram} GB RAM | {platform.system()} | CPU-only PyTorch"
print("hardware:", hw)

Xs = eng._scale(Xraw[:60000])           # pre-scaled for the pure-inference stage
eng.predict_detailed(Xraw[:512])        # warm up

def single_dist(fn_call, n=1000):
    lat = []
    for i in range(n):
        t = time.perf_counter(); fn_call(i); lat.append((time.perf_counter() - t) * 1000)
    a = np.array(lat)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3),
            "p90_ms": round(float(np.percentile(a, 90)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "max_ms": round(float(a.max()), 3)}

def batch_tput(fn_call, X, bs):
    t = time.perf_counter(); fn_call(X[:bs]); dt = time.perf_counter() - t
    return round(bs / dt, 0)

# stages: pure two-tier inference (scaled in) / full engine (raw in) / engine+adapter (DataFrame in)
df60 = pd.DataFrame(Xraw[:60000], columns=fn)
stages = {
    "two-tier inference (scaled)": (lambda i: eng.ai.predict_detailed(Xs[i:i+1]),
                                    lambda X: eng.ai.predict_detailed(eng._scale(X))),
    "full engine (rules+AI, raw)": (lambda i: eng.predict_detailed(Xraw[i:i+1]),
                                    lambda X: eng.predict_detailed(X)),
    "engine + CICFlowMeter adapter": (lambda i: RT.RealTimeIDS(eng).score_dataframe(df60.iloc[i:i+1]),
                                      lambda X: RT.RealTimeIDS(eng).score_dataframe(pd.DataFrame(X, columns=fn))),
}
rows = []
for name, (single, batch) in stages.items():
    d = single_dist(single, n=500)
    tp = batch_tput(batch, Xraw, 4096)
    rows.append({"stage": name, **d, "batch4096_flows_per_s": tp})
    print(f"  {name:32s} p50={d['p50_ms']:.3f}ms p99={d['p99_ms']:.3f}ms  batch4096={tp:,.0f}/s", flush=True)

# throughput vs batch size (pure inference)
print("\ninference throughput vs batch size:")
tp_rows = []
for bs in [1, 64, 512, 4096, 50000]:
    tp = batch_tput(lambda X: eng.ai.predict_detailed(eng._scale(X)), Xraw, bs)
    tp_rows.append({"batch_size": bs, "flows_per_s": tp}); print(f"  bs={bs:>6d}  {tp:,.0f} flows/s")

pd.DataFrame(rows).to_csv(os.path.join(TAB, "deployment_latency.csv"), index=False)
pd.DataFrame(tp_rows).to_csv(os.path.join(TAB, "deployment_throughput.csv"), index=False)
with open(os.path.join(TAB, "deployment_hardware.txt"), "w", encoding="utf-8") as f:
    f.write(hw + "\n")
print("\n>> model inference is not the bottleneck. The real capture-to-alert latency is")
print(">> dominated by the FLOW-COMPLETION FLOOR: CICFlowMeter cannot emit a flow until it")
print(">> ends (FIN/RST or the activity/flow timeout, up to 120 s), so end-to-end latency")
print(">> per flow is set by flow duration + the export interval, NOT by the ~ms inference.")
print("saved -> results/tables/deployment_{latency,throughput}.csv, deployment_hardware.txt")
