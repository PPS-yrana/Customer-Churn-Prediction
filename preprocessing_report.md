# Preprocessing Report — Customer Churn Prediction

## 1. Dataset Overview

- **Source:** `churn_data.csv`
- **Rows / columns (raw):** 500 rows, 9 columns
- **Target variable:** `Churn` (binary: 0 = retained, 1 = churned)
- **Missing values:** none found in any column
- **Class balance:** churners are a minority class (~11% of rows), so stratified
  splitting is used before modeling to keep that ratio in both train and test sets

| Column | Type | Notes |
|---|---|---|
| CustomerID | text | unique identifier, excluded from modeling |
| Tenure | numeric | months as a customer |
| MonthlyCharges | numeric | current monthly bill |
| TotalCharges | numeric | cumulative billing to date |
| Contract | categorical (ordinal) | Month-to-month / One year / Two year |
| PaymentMethod | categorical (nominal) | Credit Card / Electronic Check / Bank Transfer |
| PaperlessBilling | categorical (binary) | Yes / No |
| SeniorCitizen | numeric (binary) | already 0/1 |
| Churn | numeric (binary) | target |

## 2. Categorical Encoding (3 methods)

| Column | Method | Rationale |
|---|---|---|
| `PaperlessBilling` | **Label Encoding** | Binary column — a single 0/1 column is sufficient, no ordinality issue with only two levels |
| `PaymentMethod` | **One-Hot Encoding** | Nominal column with no inherent order; one-hot avoids implying a false ranking between payment methods |
| `Contract` | **Ordinal Encoding** | Has a genuine order (`Month-to-month` < `One year` < `Two year`) that also correlates with commitment/churn risk, so preserving the order adds signal rather than discarding it |

## 3. Feature Scaling (2 methods)

Both methods were applied to `Tenure`, `MonthlyCharges`, and `TotalCharges` so
their effects could be compared directly:

- **Min-Max scaling** — rescales each column to `[0, 1]`. Useful for models
  or visualizations that need bounded ranges, but sensitive to extreme values.
- **Standard scaling (Z-score)** — centers each column to mean 0, std 1.
  Preferred for the modeling step here since it handles the (mild) skew in
  `TotalCharges` better and is the more common choice for tree-based and
  linear models alike.

Both scaled versions are kept side by side in the dataframe (`_MinMax` /
`_Standard` suffixes) rather than overwriting the originals, so the choice of
which to feed a given model stays explicit.

## 4. Outlier Detection & Handling

Two independent detection methods were run on the three numeric columns:

- **IQR method** — flags values beyond `Q1 - 1.5*IQR` / `Q3 + 1.5*IQR`
- **Z-score method** — flags values with `|z| > 3`

**Result:** neither method found outliers in this dataset (0 flagged in both,
for all three columns). Rather than skip the step, a `cap_outliers` /
IQR-bounds-based capping function was still implemented and wired into the
final pipeline, so it will automatically winsorize any outliers that appear
in new/unseen data at inference time.

## 5. Feature Engineering

See `feature_engineering_documentation.md` for full details on the six new
features created (`CustomerLifetimeValue`, `AvgMonthlySpend`,
`PaymentEfficiency`, `TenureGroup`, `HighValueCustomer`,
`ChargesPerTenureMonth`).

## 6. Feature Selection

Two complementary techniques narrowed the candidate feature set:

1. **Correlation analysis** — each candidate feature's Pearson correlation
   with `Churn` was inspected, and the correlation matrix between features
   was checked for redundant (highly correlated) pairs.
2. **Random Forest feature importance** — a Random Forest was trained on all
   candidate features and their importances ranked; the top 8 were kept as
   `SELECTED_FEATURES` for the final model.

## 7. Complete Pipeline

All the above steps were consolidated into a single `ChurnPreprocessor` class
(scikit-learn-style `fit` / `transform` / `fit_transform` API) so that:

- Encoders and scalers are **learned only on the training split** and then
  **applied unchanged** to the test split / any new data — avoiding data
  leakage.
- The exact same encoding, scaling, outlier-capping, and feature-engineering
  logic runs identically at both training and inference time.

The pipeline was validated end-to-end: fit on an 80% stratified training
split, applied to the held-out 20% test split, and used to train a baseline
`RandomForestClassifier`, which produced a classification report and ROC-AUC
score with no errors — confirming the full pipeline runs cleanly start to
finish. Exact metric values are visible in the executed notebook, since they
depend on the random train/test split.

## 8. Files Produced

| File | Purpose |
|---|---|
| `churn_prediction_pipeline.ipynb` | Full, executed notebook (Days 1–7) |
| `churn_data.csv` | Copy of the raw dataset used |
| `preprocessing_report.md` | This report |
| `feature_engineering_documentation.md` | Detail on each engineered feature |
| `requirements.txt` | Python dependencies to reproduce the notebook |