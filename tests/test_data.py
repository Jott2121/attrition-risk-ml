"""Tests for data loading and preprocessing."""
from __future__ import annotations

import pandas as pd

from src.data import (
    CONSTANT_OR_ID_COLS,
    TARGET_COL,
    basic_clean,
    build_preprocessor,
    get_feature_types,
    load_and_prepare,
    load_raw,
    split_features_target,
)


def test_load_raw_shape():
    df = load_raw()
    assert len(df) > 1000
    assert TARGET_COL in df.columns


def test_basic_clean_drops_constants():
    df = load_raw()
    cleaned = basic_clean(df)
    for col in CONSTANT_OR_ID_COLS:
        assert col not in cleaned.columns


def test_split_features_target_produces_binary():
    df = basic_clean(load_raw())
    X, y = split_features_target(df)
    assert TARGET_COL not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert 0.10 < y.mean() < 0.25  # base rate ~16%


def test_preprocessor_transforms_without_error():
    X, y, pre = load_and_prepare()
    pre.fit(X)
    transformed = pre.transform(X)
    assert transformed.shape[0] == len(X)


def test_feature_types_partition():
    df = basic_clean(load_raw())
    X, _ = split_features_target(df)
    num, cat = get_feature_types(X)
    # Every feature column should land in exactly one bucket
    assert set(num + cat) == set(X.columns)
    assert not (set(num) & set(cat))
