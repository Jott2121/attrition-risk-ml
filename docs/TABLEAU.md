# Building the Attrition Dashboard in Tableau

The Streamlit app (`streamlit run app/streamlit_app.py`) is the primary interactive dashboard. For reviewers who prefer Tableau, this repo ships a clean, Tableau-ready CSV at `tableau/attrition_scored.csv` and the recipe below for the six views that make up an executive-grade attrition dashboard.

## Generate the CSV

```bash
python -m src.export_tableau
```

This writes `tableau/attrition_scored.csv` with one row per employee and the following columns:

| Column | Type | Purpose |
|---|---|---|
| `predicted_attrition_probability` | Number (decimal) | Measure for heatmaps and averages |
| `predicted_attrition_band` | Dimension (Low/Medium/High) | Fast filter and color |
| `actual_attrition` | Number (0/1) | For observed-vs-predicted comparison |
| `tenure_bucket` | Dimension | Pre-bucketed for easier viz |
| `comp_bucket` | Dimension | Income quintile |
| All 30 original features | Various | Level, role, department, OverTime, etc. |

## Six dashboard views to build

### 1. KPI strip (top of dashboard)
Four headline tiles:

| Tile | Measure | Config |
|---|---|---|
| Workforce size | `COUNT(Employee Number)` | Just show count |
| % at high risk | `SUM(IIF([predicted_attrition_band] = "High", 1, 0)) / COUNT([Employee Number])` | Format as % |
| Observed attrition rate | `AVG([actual_attrition])` | Format as % |
| Model avg probability | `AVG([predicted_attrition_probability])` | Format as % |

### 2. Attrition rate by job role (bar chart)
- **Rows:** `JobRole` sorted descending
- **Columns:** `AVG(predicted_attrition_probability)`
- **Color:** `predicted_attrition_band`
- Reference line at the overall average.

### 3. Risk band distribution by department (stacked bar)
- **Rows:** `Department`
- **Columns:** `COUNT(Employee Number)` with % of total computation
- **Color:** `predicted_attrition_band` (Low=green, Medium=amber, High=red)

### 4. OverTime × JobSatisfaction heatmap
- **Rows:** `OverTime`
- **Columns:** `JobSatisfaction` (1–4)
- **Color:** `AVG(predicted_attrition_probability)`
- This single view usually tells the whole story: the darkest cell is OverTime=Yes × JobSatisfaction=1.

### 5. Tenure × Comp bucket scatter
- **Columns:** `YearsAtCompany`
- **Rows:** `MonthlyIncome`
- **Size:** `COUNT(Employee Number)`
- **Color:** `AVG(predicted_attrition_probability)`
- Makes the first-3-year × low-comp quadrant visually obvious.

### 6. Top 20 flagged employees (table)
- **Rows:** `EmployeeNumber` (sorted by probability descending, Top N filter = 20)
- **Columns:** Employee details (JobRole, Department, OverTime, YearsAtCompany, MonthlyIncome)
- **Color:** `predicted_attrition_probability`

## Filters to add to the dashboard

- `Department` (multi-select)
- `JobRole` (multi-select)
- `predicted_attrition_band` (radio)
- `tenure_bucket` (multi-select)

## Notes

- **Model-predicted probability is a measure, not a fact.** Any view should clearly label the column as predicted.
- **Avoid plotting predictions against individuals by name** in shared dashboards — aggregate to role or department level unless the viewer has authorization to see individual scores.
- **Power BI and Looker work identically** — the same CSV and the same views.
