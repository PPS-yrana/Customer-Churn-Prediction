# Customer Churn Prediction

Week 10 project — **Data Preprocessing & Feature Engineering**. This repo
preprocesses raw telecom-style customer data, engineers new predictive
features, and trains a baseline model to flag customers likely to churn.

## Project Structure

```
.
├── churn_prediction_pipeline.ipynb        # Full notebook (Days 1–7)
├── churn_data.csv                         # Raw dataset (500 customers)
├── preprocessing_report.md                # Write-up of every preprocessing step
├── feature_engineering_documentation.md   # Detail on each engineered feature
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook churn_prediction_pipeline.ipynb
```

## Dataset

`churn_data.csv` — 500 customers, 9 columns: `CustomerID`, `Tenure`,
`MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`,
`PaperlessBilling`, `SeniorCitizen`, and the target `Churn` (0/1). No missing
values.

## What the Notebook Does

| Day | Step | Details |
|---|---|---|
| 1 | Explore & understand | Data types, missing values, churn class distribution |
| 2 | Categorical encoding | Label Encoding (`PaperlessBilling`), One-Hot Encoding (`PaymentMethod`), Ordinal Encoding (`Contract`) |
| 3 | Feature scaling | Min-Max and Standard (Z-score) scaling, compared side by side |
| 4 | Outlier detection | IQR method + Z-score method on all numeric columns |
| 5 | Feature engineering | 6 new features — see below |
| 6 | Feature selection | Correlation analysis + Random Forest feature importance |
| 7 | Full pipeline | A single `ChurnPreprocessor` class (`fit` / `transform`), validated end-to-end with a baseline `RandomForestClassifier` |

### Engineered Features

- `CustomerLifetimeValue` — `Tenure x MonthlyCharges`
- `AvgMonthlySpend` — `TotalCharges / Tenure`
- `PaymentEfficiency` — `TotalCharges / CustomerLifetimeValue`
- `TenureGroup` — binned lifecycle stage (New / Growing / Established / Loyal)
- `HighValueCustomer` — flag for above-median monthly spend
- `ChargesPerTenureMonth` — `MonthlyCharges / (Tenure + 1)`

Full rationale for each is in `feature_engineering_documentation.md`.

## Results

- No missing values and no IQR/Z-score-agreed outliers were found in this
  dataset.
- The `ChurnPreprocessor` pipeline was fit only on the training split and
  applied unchanged to the test split, avoiding data leakage.
- A baseline Random Forest trained on the top 8 selected features produced a
  classification report and ROC-AUC score with no pipeline errors — exact
  metrics are in the executed notebook, since they depend on the random
  train/test split.

## Code Style

Preprocessing logic is written as small, single-purpose, docstring-documented
functions (`label_encode`, `one_hot_encode`, `ordinal_encode`,
`iqr_outlier_mask`, `zscore_outlier_mask`, `cap_outliers`,
`engineer_features`), consolidated into one `ChurnPreprocessor` dataclass for
reuse — no duplicated logic between the exploratory notebook cells and the
final pipeline.
