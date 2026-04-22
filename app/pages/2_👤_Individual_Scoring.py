"""Individual attrition scoring with SHAP explanation.

Purpose: score a specific employee profile and explain why. This is the
"conversation starter" view an HRBP would use before a 1:1 — not the
portfolio view.
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
from src.explain import shap_values_for  # noqa: E402

st.set_page_config(page_title="Individual Scoring", layout="wide", page_icon="👤")

ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(ROOT / "models" / "best_model.joblib")
    name_path = ROOT / "models" / "best_model_name.joblib"
    model_name = joblib.load(name_path) if name_path.exists() else "model"
    return pipeline, model_name


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return basic_clean(load_raw())


pipeline, model_name = load_artifacts()
ref = load_reference_data()
feature_cols = [c for c in ref.columns if c != TARGET_COL]


st.title("👤 Individual Attrition Scoring")
st.caption(
    "Build an employee profile, get a predicted risk band, and see the top "
    "10 drivers behind the score. Use the **Workforce Dashboard** page for "
    "aggregate views."
)

# Sidebar ---------------------------------------------------------------------
medians = ref[feature_cols].select_dtypes(include=[np.number]).median()
modes = ref[feature_cols].select_dtypes(exclude=[np.number]).mode().iloc[0]


def _input_for(col: str):
    if col in medians.index:
        s = ref[col]
        lo, hi = float(s.min()), float(s.max())
        default = float(medians[col])
        if (s.dropna() % 1 == 0).all():
            return st.sidebar.number_input(
                col, min_value=int(lo), max_value=int(hi), value=int(default), step=1
            )
        return st.sidebar.slider(col, lo, hi, default)
    options = sorted(ref[col].dropna().unique().tolist())
    default_index = options.index(modes[col]) if modes[col] in options else 0
    return st.sidebar.selectbox(col, options, index=default_index)


PRIORITY = [
    "JobRole", "Department", "JobLevel", "Age", "MonthlyIncome",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "OverTime", "BusinessTravel", "JobSatisfaction",
    "EnvironmentSatisfaction", "WorkLifeBalance", "DistanceFromHome",
    "MaritalStatus", "TotalWorkingYears",
]
remaining = [c for c in feature_cols if c not in PRIORITY]

st.sidebar.header("Employee profile")
with st.sidebar.expander("Key drivers", expanded=True):
    primary = {c: _input_for(c) for c in PRIORITY if c in feature_cols}
with st.sidebar.expander("Other features", expanded=False):
    other = {c: _input_for(c) for c in remaining}
inputs = {**primary, **other}
go = st.sidebar.button("Score this employee", type="primary", use_container_width=True)

# --- Score + explain ---------------------------------------------------------
if go:
    X_new = pd.DataFrame([inputs])[feature_cols]
    proba = float(pipeline.predict_proba(X_new)[0, 1])

    if proba >= 0.60:
        band, color, desc = "HIGH", "#c0392b", "Priority retention conversation"
    elif proba >= 0.30:
        band, color, desc = "MEDIUM", "#d68910", "Check in with manager this quarter"
    else:
        band, color, desc = "LOW", "#1e8449", "No immediate concern"

    col1, col2, col3 = st.columns([1, 1, 2])
    col1.metric("Attrition probability", f"{proba:.0%}")
    col2.markdown(
        f"""<div style='padding: 12px; border-radius: 8px; background: {color};
        color: white; text-align: center;'>
        <div style='font-size: 12px; opacity: 0.8;'>RISK BAND</div>
        <div style='font-size: 28px; font-weight: 700;'>{band}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    col3.info(f"**Recommended action:** {desc}")

    st.divider()
    st.subheader("Why the model predicts this")
    st.caption(
        "Each feature's contribution to this specific prediction. "
        "Positive values push toward leaving; negative toward staying."
    )

    background = ref.drop(columns=[TARGET_COL]).sample(100, random_state=42)
    shap_vals = shap_values_for(pipeline, X_new, X_background=background)

    vals = np.asarray(shap_vals.values[0])
    names = list(shap_vals.feature_names)
    top_idx = np.argsort(-np.abs(vals))[:10]
    top = pd.DataFrame(
        {"feature": [names[i] for i in top_idx], "shap_value": vals[top_idx]}
    )
    top["direction"] = np.where(top["shap_value"] > 0, "↑ risk", "↓ risk")

    st.dataframe(
        top.style.format({"shap_value": "{:+.3f}"}).background_gradient(
            subset=["shap_value"], cmap="RdBu_r"
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("👈 Fill in the employee profile and click **Score this employee**.")
    st.caption(
        "Defaults are the population median / mode. Adjust any field to score "
        "a specific profile."
    )
