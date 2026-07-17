# -*- coding: utf-8 -*-
"""Deployable entry point for the two-layer IDS — connects the whole system.

Modes
-----
  --tail FILE     LIVE (real streaming): follow a CICFlowMeter live-monitor CSV
                  (<date>_Flow.csv) that is appended one flow per row, and score
                  each flow the instant it completes (tail -f). This is the real
                  capture path — CICFlowMeter sniffs the interface with jnetpcap and
                  closes flows on FIN/RST/timeout, we consume them sub-second later.
  --watch DIR     follow a folder, scoring each new *_Flow.csv file as it appears
  --csv  FILE     score one CICFlowMeter Flow.csv (offline / one-shot)
  --replay [N]    replay N labelled test flows as a live-arrival simulation

Real-time pipeline:
    CICFlowMeter live monitor (sniff NIC via jnetpcap)
        --append one row per completed flow--> <date>_Flow.csv
        --tail -f--> run_ids.py --tail  -->  Layer 1 rules -> Layer 2 two-tier AI  -->  alerts

To start the capture side: run the CICFlowMeter GUI, open the "Realtime" monitor,
select your interface, Start; it writes <date>_Flow.csv into its save folder. Then
point --tail at that file.
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from detection_engine import DetectionEngine
import realtime as RT

BASE = os.path.dirname(__file__)
PROC = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "results", "models")


def banner(mode):
    print("=" * 70)
    print(f"  Two-layer IDS  |  Layer 1 rules -> Layer 2 XGBoost + hardened DNN")
    print(f"  mode: {mode}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Two-layer deep-learning IDS")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tail", metavar="FILE", help="LIVE: follow a CICFlowMeter <date>_Flow.csv")
    g.add_argument("--watch", metavar="DIR", help="follow a folder of *_Flow.csv files")
    g.add_argument("--csv", metavar="FILE", help="score one CICFlowMeter Flow.csv (one-shot)")
    g.add_argument("--replay", nargs="?", const=5000, type=int, metavar="N",
                   help="replay N labelled test flows as a live simulation")
    ap.add_argument("--poll", type=float, default=0.5, help="live poll seconds (--tail/--watch)")
    ap.add_argument("--idle", type=float, default=None, help="stop after N idle seconds")
    ap.add_argument("--from-start", action="store_true", help="--tail: also score existing rows")
    args = ap.parse_args()

    eng = DetectionEngine(MODELS, PROC)
    ids = RT.RealTimeIDS(eng)

    if args.tail:
        banner(f"LIVE tail  {args.tail}")
        print("following CICFlowMeter live output, one alert per completed attack flow")
        print("(start CICFlowMeter's Realtime monitor on your interface first)  Ctrl+C to stop ...")
        try:
            ids.run(RT.iter_tail_csv(args.tail, poll=args.poll, from_start=args.from_start,
                                     max_idle=args.idle), status_every=1)
        except KeyboardInterrupt:
            print(f"\n[stopped] {ids.status()}")

    elif args.csv:
        banner(f"score CSV  {args.csv}")
        ids.run(RT.iter_csv(args.csv))

    elif args.watch:
        banner(f"LIVE watch  {args.watch}")
        print("waiting for new *_Flow.csv (Ctrl+C to stop) ...")
        try:
            ids.run(RT.iter_watch_dir(args.watch, poll=args.poll, max_idle=args.idle),
                    status_every=1)
        except KeyboardInterrupt:
            print(f"\n[stopped] {ids.status()}")

    else:  # replay
        banner(f"replay  {args.replay} flows")
        Xraw = np.load(os.path.join(PROC, "X_test_raw.npy"))
        y = np.load(os.path.join(PROC, "y_test.npy"))
        rng = np.random.RandomState(1)
        idx = rng.choice(len(Xraw), min(args.replay, len(Xraw)), replace=False)
        ids.run(RT.iter_replay_df(Xraw[idx], eng.feature_names, batch=512),
                status_every=2000)


if __name__ == "__main__":
    main()
