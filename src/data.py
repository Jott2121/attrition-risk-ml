"""Data loading and preprocessing for IBM HR Attrition dataset.

The IBM HR Analytics Attrition & Performance dataset contains 1,470 employees
across 35 features including demographics, compensation, job role, tenure,
satisfaction scores, and the target variable `Attrition` (Yes/No).

Source: IBM Watson Analytics sample data (publicly available).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hr_attrition.csv"

# Columns that are constants / non-informative and must be dropped before modeling.
# Verified empirically:
#   EmployeeCount is always 1
#   Over18 is always "Y"
#   StandardHours is always 80
#   EmployeeNumber is a row identifier
CONSTANT_OR_ID_COLS: list[str] = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber",
]

TARGET_COL = "Attrition"


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw IBM HR Attrition CSV and strip any BOM from the header."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant / identifier columns. Return a copy."""
    out = df.drop(columns=[c for c in CONSTANT_OR_ID_COLS if c in df.columns]).copy()
    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features X from the binary target y (1 = left, 0 = stayed)."""
    y = (df[TARGET_COL] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) using dtype heuristics."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numerics and one-hot encodes categoricals.

    Using ColumnTransformer (rather than manually encoding) keeps the full
    transformation reproducible and means the same preprocessor can be saved
    alongside the model and re-applied to new data at inference time.
    """
    numeric_cols, categorical_cols = get_feature_types(X)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def load_and_prepare(
    path: Path | str = DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """One-shot: load raw CSV, clean, split, and build a preprocessor.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (not yet transformed).
    y : pd.Series
        Binary attrition target (1 = left).
    preprocessor : ColumnTransformer
        Fit this on training data only to avoid leakage.
    """
    raw = load_raw(path)
    cleaned = basic_clean(raw)
    X, y = split_features_target(cleaned)
    preprocessor = build_preprocessor(X)
    return X, y, preprocessor


def make_pipeline(estimator) -> Pipeline:
    """Wrap a preprocessor + estimator into a single sklearn Pipeline.

    Caller must pass in a freshly-built preprocessor (because preprocessors
    can't be shared across fits).
    """
    from sklearn.base import clone

    def _factory(X: pd.DataFrame) -> Pipeline:
        pre = build_preprocessor(X)
        return Pipeline(
            steps=[
                ("preprocess", pre),
                ("model", clone(estimator)),
            ]
        )

    return _factory
