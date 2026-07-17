# -*- coding: utf-8 -*-
"""Single source of truth for the IDS acceptance criteria.

Thresholds are set a few points inside the measured actuals (recorded in the
comments, 2026-07-16) so the suite catches regressions without flapping on noise."""

SEED = 0
SLICE_PER_CLASS = 4000          # deterministic per-class cap for the fast slice
EPS_PGD = 0.1                   # adversarial strength for the robustness tests

THRESHOLDS = {
    # --- L2 clean detection (full engine) ---
    "macro_f1_min":        0.94,    # actual 0.959
    "attack_recall_min":   0.95,    # actual 0.966
    "benign_fpr_max":      0.010,   # actual 0.0021

    # per-class DETECTION recall (flow flagged as an attack at all)
    "per_class_detect_min": {
        "DoS":        0.85,         # actual 0.921 (attempted-DoS blind spot)
        "DDoS":       0.98,         # actual 1.000
        "PortScan":   0.98,         # actual 0.9998
        "BruteForce": 0.98,         # actual 1.000
        "WebAttack":  0.90,         # actual 0.985
        "Bot":        0.90,         # actual 1.000
        "Infiltration": 0.95,       # actual 0.999
    },

    # --- L2 rule layer (Layer 1) ---
    "rule_precision_attack_min": 0.95,   # PortScan .984 / DDoS 1.0 / BruteForce .993
    "rule_benign_fpr_max":       0.010,  # actual 0.00147

    # --- L3 performance / real-time SLA ---
    "batch_throughput_min":  40000,      # flows/s ; actual ~160k
    "single_latency_ms_max": 30.0,       # ms/flow one-at-a-time ; actual ~15

    # --- L4 robustness / adversarial ---
    "dnn_pgd_evasion_max":      0.12,    # @eps=0.1 ; actual ~0.017
    "two_tier_pgd_evasion_max": 0.08,    # @eps=0.1 ; actual ~0.017
}
