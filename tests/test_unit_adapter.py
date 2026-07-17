# -*- coding: utf-8 -*-
"""L1 unit — FlowAdapter: schema mapping, aliases, zero-fill, inf/nan, valid_mask."""
import numpy as np
import pandas as pd
import realtime as RT


def test_adapt_maps_named_columns_in_order(feature_names):
    ad = RT.FlowAdapter(feature_names)
    vals = np.arange(len(feature_names), dtype="float64").reshape(1, -1)
    df = pd.DataFrame(vals, columns=feature_names)
    X, meta, missing = ad.adapt(df)
    assert X.shape == (1, len(feature_names))
    assert missing == []
    assert np.allclose(X[0], vals[0])           # order preserved


def test_adapt_alias_resolution(feature_names):
    ad = RT.FlowAdapter(feature_names)
    df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    df = df.rename(columns={"Dst Port": "Destination Port"})   # a known alias
    df["Destination Port"] = 443
    X, meta, missing = ad.adapt(df)
    assert "Dst Port" not in missing                            # resolved via alias
    assert X[0, feature_names.index("Dst Port")] == 443


def test_adapt_missing_column_zero_filled_and_reported(feature_names):
    ad = RT.FlowAdapter(feature_names)
    keep = [f for f in feature_names if f != "SYN Flag Count"]
    df = pd.DataFrame(np.ones((2, len(keep))), columns=keep)
    X, meta, missing = ad.adapt(df)
    assert "SYN Flag Count" in missing
    assert (X[:, feature_names.index("SYN Flag Count")] == 0).all()


def test_adapt_neutralises_inf_and_nan(feature_names):
    ad = RT.FlowAdapter(feature_names)
    df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    df.loc[0, "Flow Bytes/s"] = np.inf
    df.loc[0, "Flow Packets/s"] = np.nan
    X, meta, missing = ad.adapt(df)
    assert np.isfinite(X).all()


def test_valid_mask_drops_zero_duration(feature_names):
    df = pd.DataFrame({"Flow Duration": [0, 5, 100, 0]})
    mask = RT.FlowAdapter.valid_mask(df)
    assert mask.tolist() == [False, True, True, False]


def test_metadata_extracted_when_present(feature_names):
    ad = RT.FlowAdapter(feature_names)
    df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    df["Src IP"] = "10.0.0.1"; df["Dst IP"] = "10.0.0.2"; df["Timestamp"] = "t0"
    X, meta, missing = ad.adapt(df)
    assert meta.iloc[0]["Src IP"] == "10.0.0.1"
    assert meta.iloc[0]["Dst IP"] == "10.0.0.2"
