"""Workforce-level attrition risk dashboard.

Purpose: give an HR leader the view they'd actually use — not a single-
employee score but an aggregate picture of where risk lives in the org,
with the filters and drill-downs they'd ask for in a review.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import load_raw, basic_clean, TARGET_COL  # noqa: E402

st.set_page_config(page_title="Workforce Dashboard", layout="wide", page_icon="📊")

ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "best_model.joblib")


@st.cache_data
def load_scored() -> pd.DataFrame:
    """Load the dataset and attach model-predicted attrition probability."""
    df = basic_clean(load_raw())
    pipe = load_model()
    X = df.drop(columns=[TARGET_COL])
    df["proba"] = pipe.predict_proba(X)[:, 1]
    df["risk_band"] = pd.cut(
        df["proba"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return df


df = load_scored()

st.title("📊 Workforce Attrition Dashboard")
st.caption(
    "Portfolio view: where is attrition risk concentrated, and by how much? "
    "Use the filters to drill into any segment."
)

# --- Filters -----------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    roles = sorted(df["JobRole"].unique().tolist())
    selected_roles = st.multiselect("Job role", roles, default=roles)
    depts = sorted(df["Department"].unique().tolist())
    selected_depts = st.multiselect("Department", depts, default=depts)
    travel = sorted(df["BusinessTravel"].unique().tolist())
    selected_travel = st.multiselect("Business travel", travel, default=travel)
    ot = ["Yes", "No"]
    selected_ot = st.multiselect("Overtime", ot, default=ot)
    min_prob, max_prob = st.slider(
        "Predicted risk range",
        0.0,
        1.0,
        (0.0, 1.0),
        step=0.05,
    )
    st.divider()
    st.caption(
        f"Showing {sum((df['JobRole'].isin(selected_roles)) & (df['Department'].isin(selected_depts)) & (df['BusinessTravel'].isin(selected_travel)) & (df['OverTime'].isin(selected_ot)) & (df['proba'].between(min_prob, max_prob))):,} / {len(df):,} employees"
    )

filtered = df[
    (df["JobRole"].isin(selected_roles))
    & (df["Department"].isin(selected_depts))
    & (df["BusinessTravel"].isin(selected_travel))
    & (df["OverTime"].isin(selected_ot))
    & (df["proba"].between(min_prob, max_prob))
].copy()

if filtered.empty:
    st.warning("No employees match the current filters.")
    st.stop()

# --- KPIs --------------------------------------------------------------------
avg_risk = filtered["proba"].mean()
high_risk_count = int((filtered["proba"] >= 0.60).sum())
high_risk_pct = high_risk_count / len(filtered)
median_tenure = filtered["YearsAtCompany"].median()
median_income = filtered["MonthlyIncome"].median()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered employees", f"{len(filtered):,}")
c2.metric("Average predicted risk", f"{avg_risk:.1%}")
c3.metric("High-risk employees", f"{high_risk_count:,}", f"{high_risk_pct:.1%}")
c4.metric("Median tenure / comp", f"{median_tenure:.0f}y / ${median_income:,.0f}")

st.divider()

# --- Risk distribution -------------------------------------------------------
col1, col2 = st.columns((2, 1))
with col1:
    st.subheader("Risk distribution across filtered population")
    hist_df = pd.DataFrame(
        {
            "Probability": filtered["proba"],
        }
    )
    st.bar_chart(
        np.histogram(filtered["proba"], bins=20, range=(0, 1))[0],
        use_container_width=True,
    )
    st.caption("Predicted attrition probability (20 bins, 0.00 → 1.00)")

with col2:
    st.subheader("Risk band mix")
    band_counts = filtered["risk_band"].value_counts().reindex(["High", "Medium", "Low"])
    st.dataframe(
        pd.DataFrame(
            {
                "Count": band_counts,
                "%": (band_counts / band_counts.sum() * 100).round(1),
            }
        ),
        use_container_width=True,
    )

st.divider()

# --- By segment --------------------------------------------------------------
st.subheader("Risk by segment")
tab1, tab2, tab3, tab4 = st.tabs(["By Job Role", "By Department", "By Overtime", "By Tenure"])


def _segment_summary(data: pd.DataFrame, col: str) -> pd.DataFrame:
    g = data.groupby(col).agg(
        employees=("proba", "size"),
        avg_risk=("proba", "mean"),
        high_risk=("proba", lambda s: (s >= 0.60).sum()),
    )
    g["pct_high_risk"] = g["high_risk"] / g["employees"]
    return g.sort_values("avg_risk", ascending=False).round(3)


with tab1:
    summary = _segment_summary(filtered, "JobRole")
    st.dataframe(
        summary.style.format(
            {"avg_risk": "{:.1%}", "pct_high_risk": "{:.1%}"}
        ).background_gradient(subset=["avg_risk"], cmap="Reds"),
        use_container_width=True,
    )

with tab2:
    summary = _segment_summary(filtered, "Department")
    st.dataframe(
        summary.style.format(
            {"avg_risk": "{:.1%}", "pct_high_risk": "{:.1%}"}
        ).background_gradient(subset=["avg_risk"], cmap="Reds"),
        use_container_width=True,
    )

with tab3:
    summary = _segment_summary(filtered, "OverTime")
    st.dataframe(
        summary.style.format(
            {"avg_risk": "{:.1%}", "pct_high_risk": "{:.1%}"}
        ).background_gradient(subset=["avg_risk"], cmap="Reds"),
        use_container_width=True,
    )
    st.caption(
        "OverTime = Yes is consistently the single strongest risk segment — "
        "exactly what an experienced HR practitioner would expect."
    )

with tab4:
    filtered["tenure_bucket"] = pd.cut(
        filtered["YearsAtCompany"],
        bins=[-0.1, 1, 3, 5, 10, 50],
        labels=["<1 yr", "1–3 yrs", "3–5 yrs", "5–10 yrs", "10+ yrs"],
    )
    summary = _segment_summary(filtered, "tenure_bucket")
    st.dataframe(
        summary.style.format(
            {"avg_risk": "{:.1%}", "pct_high_risk": "{:.1%}"}
        ).background_gradient(subset=["avg_risk"], cmap="Reds"),
        use_container_width=True,
    )
    st.caption("Classic first-3-year cliff pattern.")

st.divider()

# --- Top flagged employees + export -----------------------------------------
st.subheader("Top 20 highest-risk employees (filtered)")
top = filtered.sort_values("proba", ascending=False).head(20)
display_cols = [
    "proba",
    "JobRole",
    "Department",
    "YearsAtCompany",
    "OverTime",
    "MonthlyIncome",
    "BusinessTravel",
    "Age",
]
st.dataframe(
    top[display_cols].style.format({"proba": "{:.1%}", "MonthlyIncome": "${:,.0f}"}),
    use_container_width=True,
)

col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        label="⬇️ Download filtered data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="attrition_risk_export.csv",
        mime="text/csv",
    )
with col_b:
    st.download_button(
        label="⬇️ Download top-20 flagged employees (CSV)",
        data=top[display_cols + ["proba"]].to_csv(index=False).encode("utf-8"),
        file_name="attrition_top20_flagged.csv",
        mime="text/csv",
    )
