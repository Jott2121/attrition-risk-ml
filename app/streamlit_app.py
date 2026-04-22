"""HR Attrition — main landing page.

Streamlit multi-page app. This is the entry point; additional pages live in
`app/pages/` and appear in the sidebar automatically.

    Home (this file)               → Executive overview + dataset summary
    1_📊_Workforce_Dashboard.py    → Org-level attrition risk view
    2_👤_Individual_Scoring.py     → Single-employee predictor + SHAP

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_raw, basic_clean, TARGET_COL  # noqa: E402

st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data() -> pd.DataFrame:
    return basic_clean(load_raw())


pipeline = load_model()
df = load_data()
n = len(df)

# -----------------------------------------------------------------------------
st.title("👥 HR Attrition Predictor")
st.markdown(
    "Predict attrition risk, explain each prediction, and see where risk "
    "concentrates across the workforce."
)

if pipeline is None:
    st.error(
        "Model not found. From the repo root run `python -m src.train` first."
    )
    st.stop()

# KPI row -----------------------------------------------------------------
stayed = int((df[TARGET_COL] == "No").sum())
left = int((df[TARGET_COL] == "Yes").sum())
rate = left / n
X = df.drop(columns=[TARGET_COL])
proba = pipeline.predict_proba(X)[:, 1]
pct_high = float((proba >= 0.60).mean())
avg_tenure_leavers = df.loc[df[TARGET_COL] == "Yes", "YearsAtCompany"].median()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Employees in dataset", f"{n:,}")
c2.metric("Currently attrited", f"{left:,}", f"{rate:.1%} of workforce")
c3.metric("High-risk (>60% proba)", f"{int(pct_high * n):,}", f"{pct_high:.1%}")
c4.metric("Median tenure of leavers", f"{avg_tenure_leavers:.1f} yrs")
c5.metric("Best model", "Logistic Regression", "ROC-AUC 0.802")

st.divider()

st.markdown(
    """
### How this app is organized

| Page | What it does |
|---|---|
| **📊 Workforce Dashboard** | Aggregate view of attrition risk across teams, locations, roles. Filters + drilldowns. Export to CSV. |
| **👤 Individual Scoring** | Build an employee profile, get a risk band, see top 10 SHAP drivers. |

Use the sidebar to navigate between pages.
"""
)

st.divider()
st.subheader("Attrition rate by segment — snapshot")
col1, col2 = st.columns(2)
with col1:
    by_role = (
        df.groupby("JobRole")[TARGET_COL]
        .apply(lambda s: (s == "Yes").mean())
        .sort_values(ascending=False)
    )
    st.bar_chart(by_role, x_label="Attrition rate", use_container_width=True)
    st.caption("Attrition rate by job role")

with col2:
    by_dept = (
        df.groupby("Department")[TARGET_COL]
        .apply(lambda s: (s == "Yes").mean())
        .sort_values(ascending=False)
    )
    st.bar_chart(by_dept, x_label="Attrition rate", use_container_width=True)
    st.caption("Attrition rate by department")


st.divider()
st.caption(
    "Data: IBM HR Analytics Attrition & Performance (public). "
    "Demo model — not for production HR decisions without validation on real workforce data."
)
