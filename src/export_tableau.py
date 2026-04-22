"""Export a Tableau-ready CSV: one row per employee with predicted attrition
probability, risk band, and all original features.

Run:
    python -m src.export_tableau

Writes:
    tableau/attrition_scored.csv  — flat, wide table ready for drag-and-drop
                                    into Tableau / Power BI / Looker.

Why a denormalized CSV rather than a Tableau workbook?
    Tableau workbooks (.twb / .twbx) are XML that binds to a specific data
    source path on the reviewer's machine — they break the moment someone
    clones the repo to a different directory. A clean, self-describing CSV
    is both more portable and more honest about what the underlying data
    actually looks like.

The docs/TABLEAU.md file describes exactly which views to build from this CSV.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import basic_clean, load_raw, TARGET_COL

ROOT = Path(__file__).resolve().parents[1]
TABLEAU_DIR = ROOT / "tableau"
TABLEAU_DIR.mkdir(exist_ok=True)


def export() -> Path:
    pipe = joblib.load(ROOT / "models" / "best_model.joblib")
    df = basic_clean(load_raw())
    X = df.drop(columns=[TARGET_COL])
    df["predicted_attrition_probability"] = pipe.predict_proba(X)[:, 1]
    df["predicted_attrition_band"] = pd.cut(
        df["predicted_attrition_probability"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Medium", "High"],
    )
    df["actual_attrition"] = (df[TARGET_COL] == "Yes").astype(int)

    # Add a few derived metrics Tableau users love.
    df["tenure_bucket"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-0.1, 1, 3, 5, 10, 50],
        labels=["<1 yr", "1-3 yrs", "3-5 yrs", "5-10 yrs", "10+ yrs"],
    )
    df["comp_bucket"] = pd.qcut(
        df["MonthlyIncome"], q=5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    )

    out = TABLEAU_DIR / "attrition_scored.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
    return out


if __name__ == "__main__":
    export()
