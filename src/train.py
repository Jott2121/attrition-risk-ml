"""Train and evaluate attrition prediction models.

Run as a script:
    python -m src.train

This trains three models (Logistic Regression, Random Forest, XGBoost), prints
side-by-side metrics, and saves the best one to `models/best_model.joblib`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import build_preprocessor, load_and_prepare

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


@dataclass
class ModelResult:
    """Holds evaluation metrics for a trained model on the held-out test set."""

    name: str
    pipeline: Pipeline
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    brier: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    confusion: np.ndarray = field(repr=False)
    classification_report_str: str = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "brier": self.brier,
            "cv_roc_auc_mean": self.cv_roc_auc_mean,
            "cv_roc_auc_std": self.cv_roc_auc_std,
            "confusion": self.confusion.tolist(),
        }


def _candidates() -> dict[str, Any]:
    """Return the estimators we benchmark.

    - Logistic Regression: interpretable baseline. class_weight='balanced'
      is critical because the positive class (attrition) is only ~16%.
    - Random Forest: non-linear, robust to feature types, low-maintenance.
    - XGBoost: typically strongest on tabular data of this size.
      scale_pos_weight handles imbalance.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            # ~ (stayed / left) ≈ 5.2 -- penalize false negatives more.
            scale_pos_weight=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def evaluate(
    name: str,
    pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ModelResult:
    """Fit, score on the held-out test set, and cross-validate on train."""
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)

    result = ModelResult(
        name=name,
        pipeline=pipe,
        roc_auc=roc_auc_score(y_test, proba),
        pr_auc=average_precision_score(y_test, proba),
        f1=f1_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred),
        brier=brier_score_loss(y_test, proba),
        cv_roc_auc_mean=float(cv_auc.mean()),
        cv_roc_auc_std=float(cv_auc.std()),
        confusion=confusion_matrix(y_test, pred),
        classification_report_str=classification_report(y_test, pred, digits=3),
    )
    return result


def run() -> dict[str, ModelResult]:
    """Full training run. Returns dict of name -> ModelResult."""
    X, y, _ = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"Train: {len(X_train)} rows ({y_train.mean():.1%} positive) | "
        f"Test: {len(X_test)} rows ({y_test.mean():.1%} positive)"
    )

    results: dict[str, ModelResult] = {}
    for name, estimator in _candidates().items():
        pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        print(f"\n=== {name} ===")
        result = evaluate(name, pipe, X_train, X_test, y_train, y_test)
        print(
            f"  ROC-AUC: {result.roc_auc:.3f}  PR-AUC: {result.pr_auc:.3f}  "
            f"F1: {result.f1:.3f}  Recall: {result.recall:.3f}  "
            f"Brier: {result.brier:.3f}"
        )
        print(
            f"  CV ROC-AUC (5-fold): "
            f"{result.cv_roc_auc_mean:.3f} ± {result.cv_roc_auc_std:.3f}"
        )
        results[name] = result

    # Pick the best model by ROC-AUC on the held-out test set.
    best_name = max(results, key=lambda n: results[n].roc_auc)
    best = results[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC = {best.roc_auc:.3f})")

    # Persist best model + all metrics for downstream use.
    joblib.dump(best.pipeline, MODELS_DIR / "best_model.joblib")
    joblib.dump(best_name, MODELS_DIR / "best_model_name.joblib")
    with (REPORTS_DIR / "metrics.json").open("w") as fh:
        json.dump(
            {name: r.to_dict() for name, r in results.items()},
            fh,
            indent=2,
        )
    print(f"\nSaved model → {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved metrics → {REPORTS_DIR / 'metrics.json'}")
    return results


if __name__ == "__main__":
    run()
