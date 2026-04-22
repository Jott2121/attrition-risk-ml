"""Streamlit demo for the HR Attrition Predictor.

Run:
    streamlit run app/streamlit_app.py

What a user does:
    1. Enter / adjust employee profile (role, tenure, comp, satisfaction, etc).
    2. See predicted attrition probability with a risk band.
    3. See the top reasons *why* the model predicts what it does (via SHAP).

Why this matters for a People Analytics review:
    - Operational HR users don't read pickle files. A working demo proves the
      model is useful, not just accurate.
    - Per-person SHAP is what makes predictions actionable (you can tell a
      manager *why* someone is flagged, not just a score).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make `src` importable when streamlit is launched from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_raw, basic_clean, TARGET_COL  # noqa: E402
from src.explain import shap_values_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
NAME_PATH = ROOT / "models" / "best_model_name.joblib"

st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Cached loaders — prevents re-reading / refitting on every Streamlit rerun.
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Run `python -m src.train` first to create "
            "models/best_model.joblib."
        )
        st.stop()
    pipeline = joblib.load(MODEL_PATH)
    model_name = joblib.load(NAME_PATH) if NAME_PATH.exists() else "model"
    return pipeline, model_name


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    df = basic_clean(load_raw())
    return df


pipeline, model_name = load_artifacts()
ref = load_reference_data()
feature_cols = [c for c in ref.columns if c != TARGET_COL]


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("👥 HR Attrition Predictor")
st.markdown(
    """
    Estimate the probability that an individual employee will leave, and see
    the top drivers behind that prediction. Built on the IBM HR Analytics
    dataset (1,470 employees, 35 features).

    **Disclaimer:** This is a research/demo model. In production, attrition
    scores must never be the sole input to a people decision — they are a
    *starting point for conversation* with managers and HRBPs.
    """
)


# -----------------------------------------------------------------------------
# Sidebar: employee profile builder
# -----------------------------------------------------------------------------
st.sidebar.header("Employee profile")
st.sidebar.caption(
    "Defaults are the population median/mode. Adjust to score a specific "
    "profile. Click *Score* when ready."
)

medians = ref[feature_cols].select_dtypes(include=[np.number]).median()
modes = ref[feature_cols].select_dtypes(exclude=[np.number]).mode().iloc[0]


def _sidebar_input(col: str):
    if col in medians.index:
        s = ref[col]
        lo, hi = float(s.min()), float(s.max())
        default = float(medians[col])
        # Integer-like fields get integer steppers for readability.
        if (s.dropna() % 1 == 0).all():
            return st.sidebar.number_input(
                col,
                min_value=int(lo),
                max_value=int(hi),
                value=int(default),
                step=1,
            )
        return st.sidebar.slider(col, lo, hi, default)
    else:
        options = sorted(ref[col].dropna().unique().tolist())
        default_index = options.index(modes[col]) if modes[col] in options else 0
        return st.sidebar.selectbox(col, options, index=default_index)


# Show the *most business-relevant* fields first so people don't hunt.
PRIORITY = [
    "JobRole",
    "Department",
    "JobLevel",
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "OverTime",
    "BusinessTravel",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "DistanceFromHome",
    "MaritalStatus",
    "TotalWorkingYears",
]
remaining = [c for c in feature_cols if c not in PRIORITY]

with st.sidebar.expander("Key drivers", expanded=True):
    primary_inputs = {c: _sidebar_input(c) for c in PRIORITY if c in feature_cols}

with st.sidebar.expander("Other features", expanded=False):
    other_inputs = {c: _sidebar_input(c) for c in remaining}

inputs = {**primary_inputs, **other_inputs}

go = st.sidebar.button("Score this employee", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Prediction + explanation
# -----------------------------------------------------------------------------
if go:
    X_new = pd.DataFrame([inputs])[feature_cols]
    proba = float(pipeline.predict_proba(X_new)[0, 1])

    # Risk banding — tuneable thresholds; these are defensible defaults that
    # map to ~company attrition base rate.
    if proba >= 0.60:
        band, band_color, band_desc = "HIGH", "#c0392b", "Priority retention conversation"
    elif proba >= 0.30:
        band, band_color, band_desc = "MEDIUM", "#d68910", "Check in with manager this quarter"
    else:
        band, band_color, band_desc = "LOW", "#1e8449", "No immediate concern"

    col1, col2, col3 = st.columns([1, 1, 2])
    col1.metric("Attrition probability", f"{proba:.0%}")
    col2.markdown(
        f"""
        <div style='padding: 12px; border-radius: 8px; background: {band_color};
                    color: white; text-align: center;'>
            <div style='font-size: 12px; opacity: 0.8;'>RISK BAND</div>
            <div style='font-size: 28px; font-weight: 700;'>{band}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.info(f"**Recommended action:** {band_desc}")

    st.divider()

    # --- SHAP explanation for this single prediction ---------------------
    st.subheader("Why the model predicts this")
    st.caption(
        "SHAP values show each feature's contribution to this specific "
        "prediction. Positive (red) = pushes toward leaving; negative (blue) "
        "= pushes toward staying."
    )

    background = ref.drop(columns=[TARGET_COL]).sample(100, random_state=42)
    shap_vals = shap_values_for(pipeline, X_new, X_background=background)

    vals = np.asarray(shap_vals.values[0])
    names = list(shap_vals.feature_names)
    top_idx = np.argsort(-np.abs(vals))[:10]
    top = pd.DataFrame(
        {"feature": [names[i] for i in top_idx], "shap_value": vals[top_idx]}
    )
    top["direction"] = np.where(top["shap_value"] > 0, "Increases risk", "Decreases risk")

    st.dataframe(
        top.style.format({"shap_value": "{:+.3f}"}).background_gradient(
            subset=["shap_value"], cmap="RdBu_r"
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    # -------------------------------------------------------------------------
    # Landing state — show dataset context so reviewers see the system cold.
    # -------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Employees", f"{len(ref):,}")
    col2.metric("Features", len(feature_cols))
    base_rate = (ref[TARGET_COL] == "Yes").mean()
    col3.metric("Attrition rate", f"{base_rate:.1%}")
    col4.metric("Best model", model_name)

    st.caption(
        "👈  Use the sidebar to build an employee profile, then click "
        "**Score this employee**."
    )

    st.subheader("Attrition rate by job role")
    by_role = (
        ref.groupby("JobRole")[TARGET_COL]
        .apply(lambda s: (s == "Yes").mean())
        .sort_values(ascending=False)
        .rename("attrition_rate")
        .reset_index()
    )
    st.bar_chart(by_role.set_index("JobRole"))

st.divider()
st.caption(
    "Data: IBM HR Analytics Attrition & Performance (publicly available). "
    "Model, code, and methodology: see README on GitHub."
)
