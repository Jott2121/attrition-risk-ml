"""Generate all README / report visualizations.

Outputs PNGs under docs/ so they can be referenced from the README.

Run:
    python -m src.visualize
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

from src.data import load_and_prepare
from src.explain import shap_values_for
import shap

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def plot_attrition_rate_by_category(df: pd.DataFrame) -> None:
    """Big-picture EDA: which segments have the highest turnover?"""
    cols = [
        "JobRole",
        "Department",
        "OverTime",
        "MaritalStatus",
        "BusinessTravel",
        "EducationField",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flat, cols):
        rate = (
            df.groupby(col)["Attrition"]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values(ascending=False)
        )
        bars = ax.barh(rate.index[::-1], rate.values[::-1], color="#2c7fb8")
        ax.axvline(
            (df["Attrition"] == "Yes").mean(),
            ls="--",
            color="firebrick",
            label="Company avg",
        )
        ax.set_title(f"Attrition rate by {col}")
        ax.set_xlabel("Attrition rate")
        ax.legend(loc="lower right", fontsize=8)
        for b, v in zip(bars, rate.values[::-1]):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=8)
    fig.suptitle("Where attrition actually concentrates", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "attrition_segments.png", bbox_inches="tight")
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Compare distributions of key numeric features for stay vs leave."""
    cols = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, cols):
        for label, group in df.groupby("Attrition"):
            ax.hist(
                group[col],
                bins=30,
                alpha=0.55,
                label=f"{label} (n={len(group)})",
                density=True,
            )
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Who leaves vs who stays: numeric distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "numeric_distributions.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics_path: Path = REPORTS_DIR / "metrics.json") -> None:
    """Bar chart of ROC-AUC / PR-AUC / F1 across the three benchmarked models."""
    with metrics_path.open() as fh:
        results = json.load(fh)
    df = pd.DataFrame(results).T[
        ["roc_auc", "pr_auc", "f1", "recall", "precision"]
    ]
    ax = df.plot(kind="bar", figsize=(10, 5), rot=0, colormap="viridis")
    ax.set_title("Model comparison on held-out test set", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close()


def plot_confusion_and_curves() -> None:
    """Confusion matrix, ROC curve, and PR curve for the best model."""
    pipe = joblib.load(MODELS_DIR / "best_model.joblib")
    X, y, _ = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_test,
        y_test,
        display_labels=["Stayed", "Left"],
        cmap="Blues",
        ax=axes[0],
    )
    axes[0].set_title("Confusion matrix (test set)")
    axes[0].grid(False)

    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("ROC curve")

    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=axes[2])
    axes[2].axhline(y_test.mean(), color="k", ls="--", alpha=0.4, label="No-skill")
    axes[2].set_title("Precision-Recall curve")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "eval_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary(n_sample: int = 300) -> None:
    """SHAP beeswarm for global feature attributions + bar chart of mean |SHAP|."""
    pipe = joblib.load(MODELS_DIR / "best_model.joblib")
    X, y, _ = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_sample = X_test.sample(min(n_sample, len(X_test)), random_state=42)

    shap_values = shap_values_for(pipe, X_sample, X_background=X_train)

    # Beeswarm (global feature impact).
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP beeswarm — features driving attrition predictions", fontweight="bold")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_beeswarm.png", bbox_inches="tight")
    plt.close()

    # Mean absolute SHAP (simple ranked feature importance).
    plt.figure(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Top features by mean |SHAP|", fontweight="bold")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_bar.png", bbox_inches="tight")
    plt.close()


def run() -> None:
    df = pd.read_csv(ROOT / "data" / "hr_attrition.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    sns.set_style("whitegrid")
    print("Generating EDA figures...")
    plot_attrition_rate_by_category(df)
    plot_numeric_distributions(df)
    print("Generating model comparison chart...")
    plot_model_comparison()
    print("Generating evaluation curves...")
    plot_confusion_and_curves()
    print("Generating SHAP explanations (may take ~30s)...")
    plot_shap_summary()
    print(f"\nAll figures saved to {DOCS_DIR}/")


if __name__ == "__main__":
    run()
