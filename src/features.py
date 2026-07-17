"""
Feature analysis and selection — WM9PH-15 coursework (v2).

Provides a Random-Forest-impurity ranking of the 78 flow features and a helper
to keep the smallest set that retains a target fraction of total importance.
Unlike the original project (which simply sliced ``scaler_feature_list[:70]``,
a list completely unrelated to the importances it had just computed), the
selected feature set here is the *same* one used downstream for training and
inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def rank_features(X, y_int, feature_names, n_estimators: int = 200,
                  random_state: int = 42) -> pd.DataFrame:
    """Return a DataFrame of features sorted by Random-Forest importance."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
    )
    rf.fit(X, y_int)
    imp = rf.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["cumulative"] = df["importance"].cumsum()
    return df


def select_by_cumulative_importance(ranking: pd.DataFrame,
                                     threshold: float = 0.99) -> list[str]:
    """Keep the top features whose cumulative importance reaches ``threshold``.

    Always keeps at least the single most important feature.
    """
    keep = ranking[ranking["cumulative"] <= threshold]["feature"].tolist()
    if not keep:
        keep = [ranking.iloc[0]["feature"]]
    elif keep[-1] != ranking.iloc[len(keep) - 1]["feature"]:
        pass
    # include the first feature that crosses the threshold
    if len(keep) < len(ranking):
        keep.append(ranking.iloc[len(keep)]["feature"])
    return keep


def indices_for(feature_names_all: list[str], selected: list[str]) -> np.ndarray:
    """Map a list of selected feature names to column indices in the full set."""
    pos = {f: i for i, f in enumerate(feature_names_all)}
    return np.array([pos[f] for f in selected], dtype=int)
