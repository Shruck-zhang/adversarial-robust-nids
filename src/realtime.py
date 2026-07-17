# -*- coding: utf-8 -*-
"""Real-time channel that connects the whole IDS together.

    live traffic --(tshark)--> pcap --(CICFlowMeter jar)--> Flow.csv
                                                              |
                                        FlowAdapter (this file)
                                                              v
                                        DetectionEngine (rules -> two-tier AI)
                                                              v
                                                    alerts + rolling stats

This module is transport-agnostic: it turns a CICFlowMeter flow table (a DataFrame
or CSV) into the model's exact 66-feature raw matrix and streams it through the
``DetectionEngine``.  It provides three flow sources — a one-shot CSV, a watched
directory (the live CICFlowMeter output folder), and an array replay (to test the
full path without a capture stack).

The column adapter is deliberately tolerant: CICFlowMeter builds differ slightly in
their column names and some builds omit a few columns (Fwd/Bwd RST Flags, ICMP
Code/Type, Total TCP Flow Time on this 4.0 build); those are matched by alias where
possible and zero-filled (and reported) otherwise, so a schema drift degrades
gracefully instead of silently corrupting the input.
"""
from __future__ import annotations

import os
import glob
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


# canonical (training) feature name -> alternative names seen in CICFlowMeter CSVs
_ALIASES = {
    "Dst Port":                   ["dst port", "destination port"],
    "Total Fwd Packet":           ["total fwd packet", "total fwd packets", "tot fwd pkts"],
    "Total Length of Fwd Packet": ["total length of fwd packet", "total length of fwd packets",
                                   "totlen fwd pkts"],
    "Packet Length Min":          ["packet length min", "min packet length"],
    "FWD Init Win Bytes":         ["fwd init win bytes", "init_win_bytes_forward",
                                   "init fwd win bytes"],
    "Bwd Init Win Bytes":         ["bwd init win bytes", "init_win_bytes_backward",
                                   "init bwd win bytes"],
    "Fwd Seg Size Min":           ["fwd seg size min", "min_seg_size_forward"],
}
_META = ["Timestamp", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]


def _norm(s):
    return str(s).strip().lower()


def _safe_int(v, fallback=0.0):
    """int() that never raises on NaN/Inf/non-numeric — falls back, then to 0."""
    try:
        fv = float(v)
        if np.isfinite(fv):
            return int(fv)
    except (TypeError, ValueError):
        pass
    try:
        fb = float(fallback)
        return int(fb) if np.isfinite(fb) else 0
    except (TypeError, ValueError):
        return 0


class FlowAdapter:
    """Reindex/clean a raw CICFlowMeter DataFrame to the model's raw feature order."""

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)

    def adapt(self, df):
        """Return (X_raw float64 [n, 66], meta DataFrame, missing list)."""
        present = {_norm(c): c for c in df.columns}
        cols, missing = {}, []
        for feat in self.feature_names:
            src = None
            for alias in _ALIASES.get(feat, []):
                if alias in present:
                    src = present[alias]; break
            if src is None and _norm(feat) in present:
                src = present[_norm(feat)]
            if src is None:
                cols[feat] = 0.0; missing.append(feat)
            else:
                cols[feat] = pd.to_numeric(df[src], errors="coerce")
        out = pd.DataFrame(cols, columns=self.feature_names)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # explicit index so all-missing metadata (scalar "") broadcasts to n rows
        meta = pd.DataFrame({m: (df[present[_norm(m)]].to_numpy() if _norm(m) in present else "")
                             for m in _META}, index=df.index)
        return out.to_numpy("float64"), meta.reset_index(drop=True), missing

    @staticmethod
    def valid_mask(df):
        """Keep only real flows (positive duration), mirroring training cleaning."""
        present = {_norm(c): c for c in df.columns}
        if "flow duration" in present:
            dur = pd.to_numeric(df[present["flow duration"]], errors="coerce").fillna(0)
            return (dur > 0).to_numpy()
        return np.ones(len(df), bool)


@dataclass
class Alert:
    ts: str
    src: str
    dst: str
    dport: int
    proto: int
    verdict: str
    source: str          # rule / ai / both
    reason: str
    disagree: bool

    def line(self):
        who = f"{self.src or '?'} -> {self.dst or '?'}:{self.dport}"
        flag = "  [!] rule/AI conflict" if self.disagree else ""
        return (f"[{self.ts}] ATTACK {self.verdict:<12} {who:<34} "
                f"via={self.source:<5} {self.reason}{flag}")


class RealTimeIDS:
    """Streaming detector: DataFrame/CSV/array frames -> alerts + rolling stats."""

    def __init__(self, engine, adapter=None):
        self.engine = engine
        self.adapter = adapter or FlowAdapter(engine.feature_names)
        self.cn = engine.class_names
        self.benign = engine.benign
        self.reset_stats()

    def reset_stats(self):
        self.n_flows = 0
        self.n_attacks = 0
        self.per_class = {c: 0 for c in self.cn}
        self.n_conflict = 0
        self.warned_missing = False

    # ---- core: score a raw matrix (+ optional meta) -> alerts ---- #
    def score_matrix(self, X_raw, meta=None):
        X_raw = np.asarray(X_raw, dtype="float64")
        det = self.engine.predict_detailed(X_raw)
        alerts = []
        dport_i = self.engine.feature_names.index("Dst Port")
        proto_i = self.engine.feature_names.index("Protocol")
        for k in np.where(det["is_attack"])[0]:
            if meta is not None and len(meta):
                r = meta.iloc[k]
                ts = str(r.get("Timestamp", "")).strip() or time.strftime("%H:%M:%S")
                src, dst = str(r.get("Src IP", "")).strip(), str(r.get("Dst IP", "")).strip()
                dport = _safe_int(r.get("Dst Port", None), X_raw[k, dport_i])
                proto = _safe_int(r.get("Protocol", None), X_raw[k, proto_i])
            else:
                ts, src, dst = time.strftime("%H:%M:%S"), "", ""
                dport, proto = _safe_int(X_raw[k, dport_i]), _safe_int(X_raw[k, proto_i])
            alerts.append(Alert(ts, src, dst, dport, proto,
                                self.cn[int(det["final"][k])], str(det["source"][k]),
                                det["reason"][k] or "AI-detected anomaly (XGBoost/DNN)",
                                bool(det["disagree"][k])))
        self._update_stats(det)
        return {"alerts": alerts, "det": det, "n": len(X_raw)}

    def score_dataframe(self, df):
        valid = self.adapter.valid_mask(df)
        df = df[valid].reset_index(drop=True)
        if len(df) == 0:
            return {"alerts": [], "det": None, "n": 0}
        X, meta, missing = self.adapter.adapt(df)
        if missing and not self.warned_missing:
            print(f"[adapter] {len(missing)} feature(s) not in CSV, zero-filled: {missing}")
            self.warned_missing = True
        return self.score_matrix(X, meta)

    def _update_stats(self, det):
        self.n_flows += len(det["final"])
        atk = det["is_attack"]
        self.n_attacks += int(atk.sum())
        self.n_conflict += int(det["disagree"].sum())
        for k in np.where(atk)[0]:
            self.per_class[self.cn[int(det["final"][k])]] += 1

    def status(self):
        top = sorted(((v, c) for c, v in self.per_class.items() if v),
                     reverse=True)[:4]
        brk = "  ".join(f"{c}:{v}" for v, c in top) or "none"
        return (f"flows={self.n_flows:,}  attacks={self.n_attacks:,}  "
                f"conflicts={self.n_conflict:,}  [{brk}]")

    # ---- streaming runner ---- #
    def run(self, frame_iter, on_alert=None, status_every=0):
        """Consume an iterator of DataFrames (CSV chunks / watched files)."""
        t0 = time.time()
        for df in frame_iter:
            res = self.score_dataframe(df)
            for a in res["alerts"]:
                (on_alert or (lambda x: print("  " + x.line())))(a)
            if status_every and self.n_flows and self.n_flows % status_every < res["n"]:
                print(f"  ... {self.status()}")
        dt = time.time() - t0
        rate = self.n_flows / dt if dt > 0 else 0
        print(f"\n[done] {self.status()}  ({self.n_flows:,} flows in {dt:.1f}s, {rate:,.0f} flows/s)")


# --------------------------- flow sources --------------------------- #
def iter_csv(path):
    """One-shot: yield a single CICFlowMeter Flow.csv as a DataFrame."""
    yield pd.read_csv(path)


def iter_watch_dir(directory, pattern="*_Flow.csv", poll=2.0, max_idle=None):
    """Live: watch the CICFlowMeter output folder and yield each new flow CSV as it
    appears (pair with CICFlowMeter running on the capture interface)."""
    seen = set(glob.glob(os.path.join(directory, pattern)))
    idle = 0.0
    while True:
        new = [f for f in glob.glob(os.path.join(directory, pattern)) if f not in seen]
        for f in sorted(new, key=os.path.getmtime):
            seen.add(f)
            try:
                yield pd.read_csv(f)
            except Exception as e:
                print(f"[watch] skip {os.path.basename(f)}: {e}")
        if new:
            idle = 0.0
        else:
            time.sleep(poll); idle += poll
            if max_idle and idle >= max_idle:
                return


def iter_tail_csv(path, poll=0.5, from_start=False, max_idle=None):
    """LIVE: follow a CICFlowMeter real-time monitor CSV (``<date>_Flow.csv``) that
    is being appended to one completed flow per row, and yield each new flow as a
    single-row DataFrame the instant it is written (``tail -f`` semantics).

    This is the real streaming consumer: CICFlowMeter's live monitor sniffs the
    interface with jnetpcap, closes a flow on FIN/RST or timeout, and appends its
    feature row here — so we score exactly one flow per completed flow, sub-second
    after it ends, with the *same* extractor used to build the training data.

    ``from_start=False`` (default) skips rows already in the file and only reports
    flows that arrive after we attach, as a live sensor would.  Handles partial
    lines (writer mid-flush), a not-yet-created file, and repeated header rows.
    """
    while not os.path.exists(path):
        time.sleep(poll)
    f = open(path, "r", encoding="utf-8", errors="replace")
    header = None
    while header is None:                       # wait for the header line
        line = f.readline()
        if line and line.strip():
            header = [c.strip() for c in line.rstrip("\n").split(",")]
        else:
            time.sleep(poll)
    header_line = ",".join(header)
    if not from_start:
        f.seek(0, os.SEEK_END)                  # only flows that arrive from now on
    buf, idle = "", 0.0
    while True:
        rows = []                               # drain every complete line available now
        while True:
            line = f.readline()
            if not line:
                break
            if not line.endswith("\n"):         # writer mid-flush -> keep the partial
                buf += line
                break
            line = (buf + line).strip(); buf = ""
            if not line or line == header_line:  # skip blanks / repeated headers
                continue
            row = line.split(",")
            if len(row) >= len(header):
                rows.append(row[:len(header)])
        if rows:                                # micro-batch: one chunk per poll cycle
            idle = 0.0
            yield pd.DataFrame(rows, columns=header)
        else:
            time.sleep(poll); idle += poll
            if max_idle and idle >= max_idle:
                return


def iter_replay_df(X_raw, feature_names, batch=256, rate=None):
    """Test: replay a raw feature matrix as if arriving live, ``batch`` flows at a
    time, as DataFrames with named columns (so the full adapter path is exercised).
    ``rate`` (flows/s) throttles to simulate real timing."""
    X_raw = np.asarray(X_raw, dtype="float64")
    names = list(feature_names)
    for i in range(0, len(X_raw), batch):
        chunk = X_raw[i:i + batch]
        yield pd.DataFrame(chunk, columns=names)
        if rate:
            time.sleep(len(chunk) / rate)
