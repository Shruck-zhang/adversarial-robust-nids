# -*- coding: utf-8 -*-
"""L1 unit — RuleLayer: each signature fires on a crafted flow, boundaries hold,
priority order is respected, and an inert flow triggers nothing."""
import numpy as np
from rule_layer import RuleLayer
from conftest import make_flow


def _rl(feature_names):
    return RuleLayer(feature_names, ["Benign", "DoS", "DDoS", "PortScan", "BruteForce",
                                     "WebAttack", "Bot", "Infiltration"])


def test_portscan_signature_fires(feature_names):
    rl = _rl(feature_names)
    x = make_flow(feature_names, {"Total Fwd Packet": 1, "Total Length of Fwd Packet": 0,
                                  "Bwd Packet Length Max": 0, "SYN Flag Count": 1,
                                  "Flow Duration": 50})
    out = rl.predict_detailed(x)
    assert out["rule"][0] == "portscan"
    assert rl.class_names[out["pred"][0]] == "PortScan"


def test_ddos_signature_fires(feature_names):
    rl = _rl(feature_names)
    x = make_flow(feature_names, {"Dst Port": 80, "Total Fwd Packet": 8,
                                  "Total Length of Fwd Packet": 20, "Bwd Packet Length Max": 7000})
    out = rl.predict_detailed(x)
    assert out["rule"][0] == "ddos"
    assert rl.class_names[out["pred"][0]] == "DDoS"


def test_bruteforce_signature_fires(feature_names):
    rl = _rl(feature_names)
    x = make_flow(feature_names, {"Dst Port": 22, "Protocol": 6,
                                  "Total Fwd Packet": 11, "Down/Up Ratio": 1.5})
    out = rl.predict_detailed(x)
    assert out["rule"][0] == "bruteforce"
    assert rl.class_names[out["pred"][0]] == "BruteForce"


def test_inert_flow_triggers_nothing(feature_names):
    rl = _rl(feature_names)
    x = np.zeros((1, len(feature_names)))
    out = rl.predict_detailed(x)
    assert out["rule"][0] == ""
    assert out["pred"][0] == rl.benign


def test_portscan_boundary_too_many_packets(feature_names):
    rl = _rl(feature_names)
    # 3 fwd packets exceeds SCAN_MAX_FWD_PKTS=2 -> must NOT fire portscan
    x = make_flow(feature_names, {"Total Fwd Packet": 3, "Total Length of Fwd Packet": 0,
                                  "Bwd Packet Length Max": 0, "SYN Flag Count": 1,
                                  "Flow Duration": 50})
    out = rl.predict_detailed(x)
    assert out["rule"][0] != "portscan"


def test_bruteforce_needs_low_downup_rejected(feature_names):
    rl = _rl(feature_names)
    # down/up ratio below BRUTE_MIN_DOWNUP=1.3 -> not brute force
    x = make_flow(feature_names, {"Dst Port": 22, "Protocol": 6,
                                  "Total Fwd Packet": 11, "Down/Up Ratio": 1.0})
    out = rl.predict_detailed(x)
    assert out["rule"][0] != "bruteforce"


def test_rule_layer_is_deterministic(feature_names, slice_raw):
    rl = _rl(feature_names)
    X, _ = slice_raw
    a = rl.predict(X[:2000]); b = rl.predict(X[:2000])
    assert np.array_equal(a, b)
