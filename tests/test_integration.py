# -*- coding: utf-8 -*-
"""L2 integration — end-to-end behaviour on real data: clean-metric regression
guards, per-class detection floors, rule-layer precision, a real CICFlowMeter CSV,
and CLI smoke tests."""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from thresholds import THRESHOLDS as T


def test_clean_metrics_regression(engine, slice_raw, class_names):
    X, y = slice_raw
    d = engine.predict_detailed(X)
    ben = engine.benign
    macro_f1 = f1_score(y, d["final"], average="macro")
    attack_recall = float(d["is_attack"][y != ben].mean())
    benign_fpr = float(d["is_attack"][y == ben].mean())
    assert macro_f1 >= T["macro_f1_min"], f"macro-F1 {macro_f1:.4f}"
    assert attack_recall >= T["attack_recall_min"], f"attack recall {attack_recall:.4f}"
    assert benign_fpr <= T["benign_fpr_max"], f"benign FPR {benign_fpr:.4f}"


def test_per_class_detection_floor(engine, test_raw, class_names):
    X, y = test_raw
    d = engine.predict_detailed(X)
    ben = engine.benign
    for name, floor in T["per_class_detect_min"].items():
        c = class_names.index(name)
        m = y == c
        if not m.any():
            continue
        rate = float(d["is_attack"][m].mean())
        assert rate >= floor, f"{name} detection {rate:.4f} < {floor}"


def test_rule_layer_precision_and_false_alarm(engine, test_raw):
    X, y = test_raw
    ben = engine.benign
    rp = engine.rules.predict(X)
    fired = rp != ben
    is_atk = y != ben
    if fired.any():
        precision = float(is_atk[fired].mean())
        assert precision >= T["rule_precision_attack_min"], f"rule precision {precision:.4f}"
    benign_fpr = float((fired & ~is_atk).sum()) / max(1, int((~is_atk).sum()))
    assert benign_fpr <= T["rule_benign_fpr_max"], f"rule benign FPR {benign_fpr:.5f}"


def test_real_cicflowmeter_csv_runs(engine, paths):
    df = pd.read_csv(paths["sample"], nrows=3000)
    import realtime as RT
    ids = RT.RealTimeIDS(engine)
    res = ids.score_dataframe(df)
    assert res["n"] > 0
    # every fused label is a valid class index
    assert set(np.unique(res["det"]["final"])).issubset(set(range(len(engine.class_names))))


@pytest.mark.parametrize("args", [["--replay", "300"], ["--csv"]])
def test_cli_smoke(paths, args):
    cmd = [sys.executable, os.path.join(paths["root"], "run_ids.py")] + args
    if args[0] == "--csv":
        cmd.append(paths["sample"])
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=paths["root"], env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
