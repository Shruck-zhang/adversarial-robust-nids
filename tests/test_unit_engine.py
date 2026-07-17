# -*- coding: utf-8 -*-
"""L1 unit — DetectionEngine: scaling matches training, fusion invariants hold, and
prediction is deterministic."""
import numpy as np


def test_scaling_matches_training_transform(engine, slice_raw):
    X, _ = slice_raw
    manual = np.clip(engine.scaler.transform(X[:500]), -10, 10).astype("float32")
    assert np.allclose(engine._scale(X[:500]), manual)


def test_fusion_is_attack_is_union(engine, slice_raw):
    X, _ = slice_raw
    d = engine.predict_detailed(X)
    rule_hit = d["rule_pred"] != engine.benign
    ai_atk = np.isin(d["ai_pred"], [c for c in range(len(engine.class_names))]) & (d["ai_pred"] != engine.benign)
    # is_attack must equal (rule fired) OR (ai flagged an attack)
    assert np.array_equal(d["is_attack"], rule_hit | ai_atk)


def test_fusion_label_prefers_ai_when_ai_flags(engine, slice_raw):
    X, _ = slice_raw
    d = engine.predict_detailed(X)
    ai_atk = d["ai_pred"] != engine.benign
    # where the AI flags an attack, the fused label is the AI's sub-type
    assert np.array_equal(d["final"][ai_atk], d["ai_pred"][ai_atk])


def test_fusion_rule_only_sets_disagreement(engine, slice_raw):
    X, _ = slice_raw
    d = engine.predict_detailed(X)
    rule_hit = d["rule_pred"] != engine.benign
    ai_atk = d["ai_pred"] != engine.benign
    rule_only = rule_hit & ~ai_atk
    if rule_only.any():
        # a rule firing while the AI says benign is surfaced as a conflict, and the
        # fused label falls back to the rule's class
        assert d["disagree"][rule_only].all()
        assert np.array_equal(d["final"][rule_only], d["rule_pred"][rule_only])


def test_source_labels_consistent(engine, slice_raw):
    X, _ = slice_raw
    d = engine.predict_detailed(X)
    rule_hit = d["rule_pred"] != engine.benign
    ai_atk = d["ai_pred"] != engine.benign
    assert (d["source"][rule_hit & ai_atk] == "both").all()
    assert (d["source"][rule_hit & ~ai_atk] == "rule").all()
    assert (d["source"][~rule_hit & ai_atk] == "ai").all()
    assert (d["source"][~rule_hit & ~ai_atk] == "benign").all()


def test_prediction_is_deterministic(engine, slice_raw):
    X, _ = slice_raw
    assert np.array_equal(engine.predict(X[:3000]), engine.predict(X[:3000]))
