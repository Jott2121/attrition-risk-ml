# HR Attrition Predictor

Predicting voluntary employee attrition from standard HR data — and, more importantly, explaining *why* the model makes each prediction so HR business partners can act on it.

**Built by a former Fortune 500 talent acquisition leader applying modern ML to the problems I spent 15+ years solving by hand.**

![Model comparison](docs/model_comparison.png)

---

## The business problem

Unwanted attrition is one of the most expensive things an employer does badly. Replacement cost for an individual contributor is typically **0.5–2x annual salary**; for specialized or leadership roles it can exceed **2x**. At a 10,000-person firm with industry-average 16% turnover and a $95K average salary, a **2 percentage-point reduction in attrition** is roughly **$19M–$38M in retained value per year**.

The difficulty isn't spotting turnover *after the fact* — HR dashboards already do that. The difficulty is **spotting it early enough to intervene, with enough specificity that an HRBP can have a real conversation** rather than a generic one.

That's what this project demonstrates.

---

## What's in this repo

| Component | What it shows |
|---|---|
| `src/data.py` | Data loading, cleaning, and reproducible preprocessing (`ColumnTransformer` with scaler + one-hot encoder). |
| `src/train.py` | Trains and benchmarks three models (Logistic Regression, Random Forest, XGBoost). 80/20 stratified split + 5-fold stratified CV. |
| `src/explain.py` | SHAP-based explainability — global and per-individual feature attributions. |
| `src/visualize.py` | Generates every figure in this README from the trained model. |
| `notebooks/01_eda_and_modeling.ipynb` | Analyst-style walkthrough: EDA → modeling → SHAP → operational recommendations. |
| `app/streamlit_app.py` | Interactive demo — enter an employee profile, get a risk band and the top 10 drivers behind the score. |

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance** — 1,470 employees, 35 features spanning demographics, compensation, role, tenure, satisfaction scores, and binary attrition label. Publicly available; widely used as a benchmark for HR modeling.

- **Target**: `Attrition` (Yes/No)
- **Base rate**: ~16.1% attrition
- **Stratified 80/20 split**, seed = 42

Four constant/identifier columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`) are dropped before modeling since they carry no signal.

### Where attrition actually concentrates

![Attrition segments](docs/attrition_segments.png)

Two findings to highlight — both consistent with what any experienced HR practitioner would expect, and exactly the signals a good model should pick up on:

- **OverTime = Yes → ~31% attrition** vs ~10% for non-OT employees. This is the single sharpest split in the dataset.
- **Sales Representative, Research Scientist (early-career), Lab Technician** show role-specific risk that sits above the company average regardless of individual attributes.

![Numeric distributions](docs/numeric_distributions.png)

The numeric distributions tell the tenure / pay / age story: leavers skew younger, shorter-tenured, lower-income, and live farther from the office — the classic first-3-year cliff pattern.

---

## Model performance

Held-out test set (294 employees, 47 positives):

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** *(selected)* | **0.802** | 0.557 | 0.638 | 0.349 | 0.451 | 0.156 |
| Random Forest | 0.783 | 0.419 | 0.106 | 0.357 | 0.164 | 0.117 |
| XGBoost | 0.766 | 0.496 | 0.362 | 0.548 | 0.436 | 0.116 |

5-fold CV on training fold confirmed stability (all three ≥ 0.80 ± 0.03).

### Why logistic regression wins here

Three reasons, and I think this is the more important finding than "we beat the benchmark":

1. **With `class_weight="balanced"` and proper preprocessing, a well-specified linear model is hard to beat on a dataset this size (n=1,470).** Tree ensembles want more rows to diverge meaningfully.
2. **Recall matters more than precision** for retention work — missing someone who leaves is much costlier than flagging someone who doesn't. LR's 0.64 recall is the strongest of the three at the default threshold.
3. **Coefficients are directly interpretable by HR stakeholders**, which matters for adoption.

### Evaluation curves

![Confusion matrix, ROC, PR curves](docs/eval_curves.png)

---

## Why the model predicts what it does — SHAP

Any model used in HR must be explainable per-individual, not just in aggregate. SHAP gives both:

### Global: what drives attrition in this workforce

![SHAP beeswarm](docs/shap_beeswarm.png)

Ranked by mean absolute contribution to predictions (top features):

![SHAP bar chart](docs/shap_bar.png)

**The signal confirms what any seasoned HR practitioner would expect:**

| Feature | Direction | Interpretation |
|---|---|---|
| **OverTime = Yes** | ↑ attrition | By far the strongest single lever — burnout signal. |
| **StockOptionLevel = 0** | ↑ attrition | Absence of equity compounds with other risks. |
| **JobRole = Sales Representative** | ↑ attrition | Role-level risk independent of individual traits. |
| **YearsAtCompany (low)** | ↑ attrition | Classic first-3-year cliff. |
| **MonthlyIncome (low)** | ↑ attrition | Pay-band effect at lower levels. |
| **BusinessTravel = Frequently** | ↑ attrition | Life/travel friction. |
| **Age (younger)** | ↑ attrition | Confounded with tenure; still significant. |

### Per-individual: why *this* employee

The Streamlit app surfaces per-prediction SHAP so an HRBP can see *"this employee is flagged because: OverTime = Yes (+), 2 years since last promotion (+), JobSatisfaction = Low (+), below-median comp for role/level (+)"* — a conversation starter, not a black box.

---

## Interactive dashboard

A multi-page Streamlit app ships with the repo:

| Page | Purpose |
|---|---|
| **Home** | Workforce summary + KPIs |
| **📊 Workforce Dashboard** | Aggregate attrition risk — filters by role, department, overtime, risk band; drill-downs by role/dept/tenure; exportable CSV of top-flagged employees |
| **👤 Individual Scoring** | Build an employee profile, get a risk band, see top 10 SHAP drivers |

```bash
pip install -r requirements.txt
python -m src.train        # train models (~30 seconds)
python -m src.visualize    # regenerate docs/ figures
streamlit run app/streamlit_app.py
```

**Tableau / Power BI / Looker users:** generate a Tableau-ready CSV:
```bash
python -m src.export_tableau   # writes tableau/attrition_scored.csv
```
Then follow [docs/TABLEAU.md](docs/TABLEAU.md) for the recipe to build six dashboard views from the exported CSV.

### Run the notebook

```bash
jupyter lab notebooks/01_eda_and_modeling.ipynb
```

---

## How I'd use this in practice

I've sat on the hiring side of thousands of requisitions. Predictions alone don't retain anyone. A model like this creates value only when it's wired into a workflow:

| Use case | How the score is used |
|---|---|
| **Manager 1:1 prep** | Surface an employee's top 3 SHAP drivers to enable a specific conversation, not a generic "stay interview." |
| **Org-level intervention** | Compare SHAP distributions across teams — flag teams where `OverTime` is the systemic driver and route to the HRBP + team lead. |
| **Retention investment prioritization** | Rank high-value / high-risk employees and estimate dollar impact of moving them from high- to medium-risk. |
| **Compensation equity cross-check** | Low-income × high-risk × under-promoted employees feed into the comp-equity audit (see [compensation-equity-analysis](https://github.com/Jott2121/compensation-equity-analysis) when published). |

**Explicitly not appropriate:**

- Adverse employment decisions (PIPs, RIFs) based on the score. Bias risk is real on HR data — this is decision *support*, not decision *making*.
- Surfacing individual scores to employees without HR and legal review.
- Using the probability without the SHAP explanation alongside it.

---

## Repo layout

```
hr-attrition-predictor/
├── data/hr_attrition.csv        # IBM HR Analytics dataset (1,470 rows)
├── src/
│   ├── data.py                  # Loading, cleaning, preprocessing pipeline
│   ├── train.py                 # Train & benchmark LR / RF / XGBoost
│   ├── explain.py               # SHAP wrapper (works with any sklearn Pipeline)
│   └── visualize.py             # Generate all README figures
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── app/streamlit_app.py         # Interactive demo
├── docs/                        # Generated PNGs referenced in this README
├── models/best_model.joblib     # Serialized best model (after training)
├── reports/metrics.json         # Full evaluation metrics
└── requirements.txt
```

---

## About

I'm **Jeff Otterson**, a talent acquisition leader with Fortune 500 experience at Amazon and Oracle. I'm building a portfolio of people analytics projects that apply modern data science and ML to the operational problems I've seen firsthand — attrition modeling, compensation equity, hiring funnel analytics, workforce planning, and responsible AI in HR.

- **LinkedIn**: [linkedin.com/in/jeffotterson](https://www.linkedin.com/in/jeffotterson/) *(update with your actual LinkedIn)*
- **MeritForge AI**: [meritforgeai.com](https://www.meritforgeai.com) — free AI career tools and research
- **This repo**: MIT licensed, contributions welcome

---

## Credits

- Dataset: IBM Watson Analytics (public). Used under standard fair-use for research/demonstration.
- Libraries: `scikit-learn`, `xgboost`, `shap`, `pandas`, `streamlit`.
