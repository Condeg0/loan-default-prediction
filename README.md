# Loan Default Prediction

Predicting loan defaults at origination using real-world LendingClub data. This project demonstrates an end-to-end machine learning pipeline to inform risk-based decision-making in consumer lending.

---

## Problem Context

Online lenders face growing challenges from rising charge-off rates. Identifying risky loans *before* approval enables better underwriting decisions, improving profitability and portfolio quality.

---

## Objective

Build a supervised learning model to predict whether a borrower will default, using features known at the time of loan origination. The goal is to support **risk-based approval or pricing**.

---

## Dataset

- **Source**: [Kaggle - Loan Data](https://www.kaggle.com/datasets/itssuru/loan-data)
- **Observations**: ~9,578 loans
- **Default rate**: ~16%

---

## Methodology

1. **Feature Engineering**:
   - Selected key variables like `credit_policy`, `FICO`, `purpose`, and `revol_util`
   - One-hot encoded categorical features
   - Scaled numerical variables using `ColumnTransformer`

2. **Modeling Pipeline**:
   - Compared classifiers: `Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost`
   - Tuned hyperparameters via 5-fold cross-validated `ROC-AUC`
   - Handled class imbalance with `class_weight='balanced'`
   - Evaluated on a held-out test set

---

## Results

- **ROC-AUC**: AUC of 0.656 on the held-out test set confirms model performance beyond baseline (0.5)
- **Recall on defaults**: 58%
- **Approval rate maintained**: 62%
- **Estimated savings**: ~$850K per 10,000 loans

- ![ROC AUC](/reports/figures/roc_test.png | width=30)

---

## Key Risk Drivers

- `credit_policy`
- `purpose = debt_consolidation`
- `FICO < 660`
- `high revolving utilization`

---

## Business Impact

Used as a **second-look model**:
- Automatically reject top 10% most risky applications
- Route mid-risk loans to manual review
- Support pricing for near-prime borrowers

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core programming |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Model training, pipeline, metrics |
| `XGBoost` | Gradient boosting |
| `matplotlib`, `seaborn` | Visualizations |
| `ColumnTransformer`, `Pipeline` | Preprocessing |

---
