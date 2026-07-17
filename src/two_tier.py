# -*- coding: utf-8 -*-
"""Two-tier detector: XGBoost (primary, best clean detection + lowest false alarm)
+ adversarially-hardened DNN (robustness backstop).

Defence-in-depth rationale: the tree is non-differentiable (gradient FGSM/PGD
cannot be mounted against it), the DNN is adversarially trained — so an attacker
must fool *both* at once. A flow is flagged as an attack when *either* tier flags
it, and a disagreement between the tiers is surfaced as a possible-manipulation
signal.
"""
from __future__ import annotations

import os
import numpy as np
import joblib


class TwoTierDetector:
    def __init__(self, models_dir, proc_dir):
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        import models_extra as MX
        self.xgb = joblib.load(os.path.join(models_dir, "XGBoost.pkl"))
        self.dnn = MX.load_dnn(os.path.join(models_dir, "dnn_hardened.pt"))
        self.class_names = list(joblib.load(os.path.join(proc_dir, "class_names.pkl")))
        self.benign = self.class_names.index("Benign")

    def predict(self, X):
        """Fused class labels (int)."""
        return self.predict_detailed(X)["final"]

    def predict_detailed(self, X):
        """Return the per-tier verdicts, the fused verdict, the attack flag and the
        tier-disagreement (possible-adversarial-manipulation) flag."""
        xgb = self.xgb.predict(X).astype(int)
        dnn = self.dnn.predict(X).astype(int)
        xgb_atk = xgb != self.benign
        dnn_atk = dnn != self.benign
        # flag as attack if EITHER tier flags it (defence in depth)
        final = np.where(xgb_atk, xgb, np.where(dnn_atk, dnn, self.benign))
        return {
            "xgb": xgb, "dnn": dnn, "final": final,
            "is_attack": xgb_atk | dnn_atk,
            "disagree": xgb != dnn,               # possible evasion / manipulation
        }
