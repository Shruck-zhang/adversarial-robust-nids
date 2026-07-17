# -*- coding: utf-8 -*-
"""Two-layer detection engine — the deployable core of the IDS.

Pipeline (see the architecture figure):
    raw CICFlowMeter flow features
        -> Layer 1  RuleLayer          (fast, deterministic, explainable; known attacks)
        -> Layer 2  TwoTierDetector    (XGBoost + hardened DNN; novel / obfuscated / adversarial)

Design contract
---------------
* The engine's input is the RAW physical feature values (same 66 columns, same
  order as ``feature_names``) that CICFlowMeter produces.  The AI layer needs the
  training-time transform, so the engine applies ``clip(scaler.transform(x), ±10)``
  internally — callers never scale by hand.
* Layer 1 may only *assert* "this is a known attack of type T"; it may NOT clear a
  flow as benign (rules encode known-bad only).  Every flow therefore also reaches
  Layer 2, which is the actual benign/attack classifier.
* Fusion is defence-in-depth OR-logic: a flow is an attack if *either* layer flags
  it.  When the rule layer fires we keep its (explainable, high-precision) label;
  otherwise we use the AI verdict.  A rule-vs-AI disagreement is surfaced as a
  possible-manipulation signal.
"""
from __future__ import annotations

import os
import numpy as np
import joblib


class DetectionEngine:
    def __init__(self, models_dir, proc_dir, clip=10.0):
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        import two_tier
        from rule_layer import RuleLayer

        self.scaler = joblib.load(os.path.join(proc_dir, "scaler.pkl"))
        self.feature_names = list(joblib.load(os.path.join(proc_dir, "feature_names.pkl")))
        self.class_names = list(joblib.load(os.path.join(proc_dir, "class_names.pkl")))
        self.benign = self.class_names.index("Benign")
        self.clip = clip
        self.rules = RuleLayer(self.feature_names, self.class_names)
        self.ai = two_tier.TwoTierDetector(models_dir, proc_dir)

    # -- turn raw physical features into the AI layer's scaled input -- #
    def _scale(self, Xraw):
        return np.clip(self.scaler.transform(Xraw), -self.clip, self.clip).astype("float32")

    def predict(self, Xraw):
        """Fused class labels (int)."""
        return self.predict_detailed(Xraw)["final"]

    def predict_detailed(self, Xraw):
        Xraw = np.asarray(Xraw, dtype="float64")
        rl = self.rules.predict_detailed(Xraw)          # raw units
        ai = self.ai.predict_detailed(self._scale(Xraw))  # scaled units

        rule_pred = rl["pred"]
        rule_hit = rule_pred != self.benign
        ai_final = ai["final"]
        ai_atk = ai["is_attack"]

        # Defence-in-depth: a flow is an attack if EITHER layer flags it. For the
        # sub-type label, trust the AI when it also flags an attack (it discriminates
        # attack families better); fall back to the rule's label only when the rule
        # fires but the AI missed it (e.g. an evaded / novel case the rule caught).
        is_attack = rule_hit | ai_atk
        final = np.where(ai_atk, ai_final, np.where(rule_hit, rule_pred, self.benign))

        source = np.where(rule_hit & ai_atk, "both",
                  np.where(rule_hit, "rule",
                  np.where(ai_atk, "ai", "benign"))).astype(object)
        # possible manipulation: a deterministic rule fired on known-attack physics
        # but the AI cleared the flow as benign (candidate AI evasion), or the two
        # AI tiers disagree.  (A rule staying silent on an AI-only attack is NOT a
        # conflict — rules simply don't cover every class.)
        disagree = (rule_hit & ~ai_atk) | ai["disagree"]

        return {
            "final": final, "is_attack": is_attack, "source": source,
            "rule_pred": rule_pred, "rule_name": rl["rule"], "reason": rl["reason"],
            "ai_pred": ai_final, "xgb": ai["xgb"], "dnn": ai["dnn"],
            "disagree": disagree,
        }

    def explain(self, Xraw, i):
        """Human-readable one-line verdict for flow ``i`` (for alerts / demo)."""
        d = self.predict_detailed(np.asarray(Xraw, dtype="float64")[i:i + 1])
        cls = self.class_names[int(d["final"][0])]
        src = d["source"][0]
        if src == "benign":
            return f"flow {i}: BENIGN (no rule fired, AI tiers agree benign)"
        reason = d["reason"][0] or "AI-detected anomaly"
        flag = "  [!] rule/AI disagreement — possible evasion" if d["disagree"][0] else ""
        return f"flow {i}: ATTACK={cls}  via={src}  reason={reason}{flag}"
