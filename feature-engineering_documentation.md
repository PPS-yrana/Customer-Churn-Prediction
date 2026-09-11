# Feature Engineering Documentation

Six new features were derived from the raw columns. Each is implemented in the
`engineer_features()` function (also embedded in `ChurnPreprocessor.transform`)
so the same logic is guaranteed to run identically on training and new data.

---

### 1. `CustomerLifetimeValue`
```
CustomerLifetimeValue = Tenure * MonthlyCharges
```
**Why:** approximates the total value a customer is expected to generate given
how long they've stayed and what they currently pay per month. Higher values
often correlate with more entrenched, lower-churn-risk customers.

### 2. `AvgMonthlySpend`
```
AvgMonthlySpend = TotalCharges / Tenure
```
**Why:** the customer's *actual* historical average monthly spend, which can
differ from their *current* `MonthlyCharges` if their plan or usage changed
over time. A widening gap between this and `MonthlyCharges` can be a churn
signal (e.g. a recent price increase).

### 3. `PaymentEfficiency`
```
PaymentEfficiency = TotalCharges / CustomerLifetimeValue
```
**Why:** a ratio comparing actual cumulative billing to the "expected" billing
implied by current tenure and rate. Values far from 1.0 indicate the
customer's plan/rate has changed significantly during their tenure, which can
flag instability in the relationship. Division-by-zero cases (new customers)
are filled with 0.

### 4. `TenureGroup`
```
Bins: 0–12 -> "New", 12–24 -> "Growing", 24–48 -> "Established", 48–72 -> "Loyal"
```
**Why:** converts a continuous tenure value into lifecycle stages. Churn risk
often isn't linear with tenure — it can be sharply elevated in the first year
and then level off — and binning lets a model pick up on that kind of
non-linear pattern more easily. The category is label-encoded
(`TenureGroup_LabelEnc`) for use in modeling.

### 5. `HighValueCustomer`
```
HighValueCustomer = 1 if MonthlyCharges > median(MonthlyCharges) else 0
```
**Why:** a simple binary flag separating above- and below-median spenders.
High-value customers may have both more to lose by leaving (loyalty programs,
sunk cost) and more incentive to churn if they feel overcharged, so this flag
lets the model treat that segment distinctly rather than relying purely on
the continuous charge amount.

### 6. `ChargesPerTenureMonth`
```
ChargesPerTenureMonth = MonthlyCharges / (Tenure + 1)
```
**Why:** normalizes current monthly charges by how long the customer has been
with the company (offset by 1 to avoid division by zero for brand-new
customers). This highlights customers paying a lot relative to how new their
relationship is — a potential early-churn risk pattern distinct from
`AvgMonthlySpend`, which looks backward at historical billing instead.

---

## Design Notes

- All six features are computed from columns already present in the raw
  dataset — no external data was joined in.
- Every feature is implemented as pure pandas/numpy vector operations (no
  row-wise `.apply()`), keeping the pipeline fast and readable.
- `engineer_features()` always returns a new DataFrame (no in-place mutation
  of the input), which keeps the function side-effect-free and safe to call
  multiple times (e.g. once per fold during cross-validation).