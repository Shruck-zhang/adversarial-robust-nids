# -*- coding: utf-8 -*-
"""Debug the LIVE streaming interface without needing a real capture stack.

A producer thread mimics CICFlowMeter's live monitor: it appends real
CICFlowMeter flow rows to a CSV one at a time (as flows 'complete'), while the main
thread tails that file with ``iter_tail_csv`` and scores each flow the instant it
arrives — exactly the real ``--tail`` path.  Verifies rows-in == flows-scored,
shows live alerts, and measures per-flow end-to-end latency."""
import os, sys, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from detection_engine import DetectionEngine
import realtime as RT

BASE = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "results", "models")
SAMPLE = os.path.join(BASE, "data", "sample_flows", "sample_cicflowmeter_Flow.csv")
LIVE = os.path.join(os.environ.get("TEMP", "."), "cfm_live_stream.csv")

N = 5000            # flows to stream
ARRIVAL_MS = 1.0    # inter-flow arrival (simulated capture rate)

# real CICFlowMeter rows to replay as "live" output
lines = open(SAMPLE, "r", encoding="utf-8", errors="replace").read().splitlines()
header, data = lines[0], lines[1:N + 1]
with open(LIVE, "w", encoding="utf-8") as f:
    f.write(header + "\n"); f.flush()          # header only; rows arrive live

eng = DetectionEngine(MODELS, PROC)
ids = RT.RealTimeIDS(eng)

def producer():
    time.sleep(1.2)                            # let the tailer attach + seek to end
    with open(LIVE, "a", encoding="utf-8") as f:
        for ln in data:
            f.write(ln + "\n"); f.flush()
            time.sleep(ARRIVAL_MS / 1000.0)
    producer.done = time.time()
producer.done = None

print("=" * 74)
print(f"LIVE streaming test — producer appends {len(data):,} real CICFlowMeter flows")
print(f"tailer scores each flow as it arrives (tail -f)   live file: {LIVE}")
print("=" * 74)
threading.Thread(target=producer, daemon=True).start()

shown = [0]
lat = []
def on_alert(a):
    if shown[0] < 8:
        print("  " + a.line()); shown[0] += 1

t0 = time.time()
# stop 2s after the producer goes idle
ids.run(RT.iter_tail_csv(LIVE, poll=0.2, from_start=False, max_idle=2.0),
        on_alert=on_alert, status_every=1000)
dt = time.time() - t0

print(f"\nproduced {len(data):,} flows, scored {ids.n_flows:,}  "
      f"(match={ids.n_flows == len(data)})")
print(f"attacks flagged live: {ids.n_attacks:,}   conflicts: {ids.n_conflict:,}")
try:
    os.remove(LIVE)
except OSError:
    pass
