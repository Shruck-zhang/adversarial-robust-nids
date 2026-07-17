# -*- coding: utf-8 -*-
"""Layer 1 — rule-based detection.

A fast, deterministic, fully-explainable signature layer that runs *before* the AI
tier.  It encodes the logic of the attacks that have a clean single-flow signature
on the CICIDS2017 flow features (port scanning, FTP/SSH brute forcing, and the
fixed-payload HTTP DDoS flood), as thresholds on the raw CICFlowMeter features.

Contract: a rule may only *assert* a known attack (high precision); it never clears
a flow as benign.  Non-matching flows fall through to the AI layer.  Thresholds are
tuned toward precision (few false alarms).

Design note (honest scope): on the *corrected* CICIDS2017 the loud "high packet
rate" behaviour actually belongs to port scans / infiltration probes, not to the
DoS/DDoS families (their flooding is slow-rate or completed-HTTP and overlaps with
benign web traffic).  Volumetric DoS and the stealthy WebAttack/Bot/Infiltration
classes are therefore left to the AI layer — which is exactly the division of
labour the two-layer design is built on.  Thresholds come from the benign-vs-attack
distributions of the train/val split (``experiments/export_raw_test.py``) and are
frozen here as auditable constants.
"""
from __future__ import annotations

import numpy as np


class RuleLayer:
    # --- thresholds in raw CICFlowMeter units (set from val distributions) --- #
    SCAN_MAX_FWD_PKTS = 2         # a probe is a single (retried) packet
    SCAN_MAX_DURATION = 2_000     # microseconds  (PortScan p95 ~ 100us)
    BRUTE_PORTS = (21, 22)        # FTP / SSH
    BRUTE_MIN_FWD_PKTS = 5
    BRUTE_MIN_DOWNUP = 1.3        # brute-force p50 ~ 1.55 vs benign ~ 1.0
    DDOS_PORT = 80
    DDOS_MIN_FWD_PKTS = 6
    DDOS_MAX_FWD_BYTES = 40       # LOIC flood: fixed ~20-byte forward payload
    DDOS_MIN_BWD_MAX = 3_000      # large server responses

    def __init__(self, feature_names, class_names):
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.idx = {f: i for i, f in enumerate(self.feature_names)}
        self.benign = self.class_names.index("Benign")
        self.cls = {c: self.class_names.index(c) for c in self.class_names}

    def _col(self, X, name):
        return X[:, self.idx[name]]

    # each rule returns a boolean mask over rows -------------------------------- #
    def _rule_portscan(self, X):
        return ((self._col(X, "Total Fwd Packet") <= self.SCAN_MAX_FWD_PKTS)
                & (self._col(X, "Total Length of Fwd Packet") == 0)   # no payload
                & (self._col(X, "Bwd Packet Length Max") == 0)        # no response
                & (self._col(X, "SYN Flag Count") >= 1)
                & (self._col(X, "Flow Duration") <= self.SCAN_MAX_DURATION))

    def _rule_bruteforce(self, X):
        dport = self._col(X, "Dst Port")
        onport = np.zeros(len(X), bool)
        for p in self.BRUTE_PORTS:
            onport |= (dport == p)
        return (onport
                & (self._col(X, "Protocol") == 6)
                & (self._col(X, "Total Fwd Packet") >= self.BRUTE_MIN_FWD_PKTS)
                & (self._col(X, "Down/Up Ratio") >= self.BRUTE_MIN_DOWNUP))

    def _rule_ddos(self, X):
        return ((self._col(X, "Dst Port") == self.DDOS_PORT)
                & (self._col(X, "Total Fwd Packet") >= self.DDOS_MIN_FWD_PKTS)
                & (self._col(X, "Total Length of Fwd Packet") <= self.DDOS_MAX_FWD_BYTES)
                & (self._col(X, "Bwd Packet Length Max") >= self.DDOS_MIN_BWD_MAX))

    def predict_detailed(self, Xraw):
        """Return per-flow rule verdict.

        Rules are applied in priority order; the first that fires wins.  Output:
          pred   : int class index (Benign where no rule fired)
          rule   : str rule name ('' where none)
          reason : str human-readable reason ('' where none)
        """
        X = np.asarray(Xraw, dtype="float64")
        n = len(X)
        pred = np.full(n, self.benign, dtype=int)
        rule = np.array([""] * n, dtype=object)
        reason = np.array([""] * n, dtype=object)

        ordered = [
            ("portscan", self._rule_portscan(X), self.cls["PortScan"],
             "single-SYN probe: <=2 fwd pkts, no payload, no response, sub-ms flow"),
            ("bruteforce", self._rule_bruteforce(X), self.cls["BruteForce"],
             "FTP/SSH login guessing: port 21/22, repeated exchange, high down/up ratio"),
            ("ddos", self._rule_ddos(X), self.cls["DDoS"],
             "HTTP flood: port 80, many small fixed-payload fwd pkts, large responses"),
        ]
        unset = np.ones(n, bool)
        for name, mask, cls, why in ordered:
            hit = mask & unset
            pred[hit] = cls
            rule[hit] = name
            reason[hit] = why
            unset &= ~hit
        return {"pred": pred, "rule": rule, "reason": reason}

    def predict(self, Xraw):
        return self.predict_detailed(Xraw)["pred"]
