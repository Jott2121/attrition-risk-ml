"""SHAP-based explainability for attrition model.

Works with any sklearn Pipeline(preprocess -> model). Uses shap.Explainer which
auto-picks TreeExplainer for tree models and LinearExplainer for logistic
regression, producing proper per-sample and global attributions either way.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _transform(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Run the pipeline's preprocessor so SHAP sees the final feature matrix."""
    pre = pipeline.named_steps["preprocess"]
    X_trans = pre.transform(X)
    # sparse -> dense for SHAP
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    feature_names = list(pre.get_feature_names_out())
    return np.asarray(X_trans), feature_names


def build_explainer(pipeline, X_background: pd.DataFrame, n_background: int = 100):
    """Create a SHAP explainer appropriate for the fitted model."""
    X_bg_trans, feature_names = _transform(pipeline, X_background.sample(
        min(n_background, len(X_background)), random_state=42
    ))
    model = pipeline.named_steps["model"]
    try:
        # Works for tree ensembles.
        explainer = shap.Explainer(model, X_bg_trans, feature_names=feature_names)
    except Exception:
        # Fallback for linear models etc.
        explainer = shap.Explainer(
            model.predict_proba, X_bg_trans, feature_names=feature_names
        )
    return explainer, feature_names


def shap_values_for(
    pipeline, X: pd.DataFrame, X_background: pd.DataFrame | None = None
) -> shap.Explanation:
    """Compute SHAP values for the rows in `X`.

    Returns a shap.Explanation containing `.values`, `.base_values`,
    and `.feature_names` — ready to plot.
    """
    if X_background is None:
        X_background = X
    explainer, feature_names = build_explainer(pipeline, X_background)
    X_trans, _ = _transform(pipeline, X)
    shap_values = explainer(X_trans)

    # For binary classifiers that return (n, m, 2), slice to the positive class.
    if shap_values.values.ndim == 3:
        shap_values = shap.Explanation(
            values=shap_values.values[..., 1],
            base_values=shap_values.base_values[..., 1]
            if shap_values.base_values.ndim > 1
            else shap_values.base_values,
            data=shap_values.data,
            feature_names=feature_names,
        )
    else:
        shap_values.feature_names = feature_names
    return shap_values


def load_best_model():
    """Convenience loader for the pipeline saved by src.train.run()."""
    return joblib.load(MODELS_DIR / "best_model.joblib")
