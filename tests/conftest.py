# -*- coding: utf-8 -*-
"""Shared fixtures: load the engine and the test data once per session, and build a
deterministic per-class slice for the fast checks."""
import os
import sys
import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from thresholds import SEED, SLICE_PER_CLASS  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "results", "models")
SAMPLE = os.path.join(ROOT, "data", "sample_flows", "sample_cicflowmeter_Flow.csv")


@pytest.fixture(scope="session")
def paths():
    return {"proc": PROC, "models": MODELS, "sample": SAMPLE, "root": ROOT}


@pytest.fixture(scope="session")
def engine(paths):
    from detection_engine import DetectionEngine
    return DetectionEngine(paths["models"], paths["proc"])


@pytest.fixture(scope="session")
def feature_names(engine):
    return list(engine.feature_names)


@pytest.fixture(scope="session")
def class_names(engine):
    return list(engine.class_names)


@pytest.fixture(scope="session")
def test_raw(paths):
    X = np.load(os.path.join(paths["proc"], "X_test_raw.npy"))
    y = np.load(os.path.join(paths["proc"], "y_test.npy"))
    return X, y


@pytest.fixture(scope="session")
def slice_raw(test_raw, class_names):
    """Deterministic per-class capped slice (fast, balanced)."""
    X, y = test_raw
    rng = np.random.RandomState(SEED)
    parts = []
    for c in range(len(class_names)):
        idx = np.where(y == c)[0]
        if len(idx) > SLICE_PER_CLASS:
            idx = rng.choice(idx, SLICE_PER_CLASS, replace=False)
        parts.append(idx)
    sel = np.concatenate(parts)
    return X[sel], y[sel]


def make_flow(feature_names, cols):
    """Build a single raw feature vector (all zeros = inert/benign baseline) with the
    given ``{exact feature name: value}`` set — used to craft deterministic flows for
    the rule unit tests (exact names handle 'Down/Up Ratio', 'Flow Packets/s', ...)."""
    x = np.zeros((1, len(feature_names)), dtype="float64")
    idx = {f: i for i, f in enumerate(feature_names)}
    for name, val in cols.items():
        x[0, idx[name]] = val
    return x
