# DP9: Loan Outstanding

## Summary

Loan Outstanding measures the farmer's existing debt burden. Outstanding loans from formal or informal sources directly reduce the farmer's capacity to take on and repay new credit. Farmers with no existing loans score highest, while those with large unpaid balances score lowest. This data point captures the fundamental lending principle that existing debt obligations compete with new loan repayment for the farmer's limited cash flow.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: Low (0.25)
- **Framework Design Weight**: 2.5
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 2.5 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP9 is scored based on whether the farmer has an outstanding loan and, if so, the amount:

```
IF(No_Outstanding_Loan, 10, IF(Outstanding_Amount <= 10000, 5, 0))
```

The raw score is then weighted by the reliability factor (0.25), resulting in an effective maximum of 2.5 points.

## Scoring Tiers

| Condition | Raw Score | Weighted Score | Rationale |
|-----------|-----------|---------------|-----------|
| No outstanding loan | 10 | 2.5 | No competing debt obligations; full repayment capacity |
| Outstanding loan ≤ 10,000 ETB | 5 | 1.25 | Manageable existing debt; partial capacity remains |
| Outstanding loan > 10,000 ETB | 0 | 0.0 | Significant debt burden; high risk of over-indebtedness |

## Survey Questions

- **Q37**: Have you had access to credit in the last 12 months? (Yes/No)
- **Q38**: Have you experienced difficulty repaying a loan? (Yes/No)
- **Q39**: Do you currently have an outstanding loan? (Yes/No)
- **Q40**: What is the total outstanding loan amount? (Numeric, range 0–100,000 ETB)

## Database Fields

- `hasAccessToCreditSource` — Whether the farmer accessed credit in the last 12 months (boolean)
- `receivedFromMicrofinance` — Whether the credit was from a microfinance institution or bank (boolean)
- `amountBorrowed` — Total loan amount in ETB
- `unableToRepayCreditOnTime` — History of repayment difficulty (boolean)
- `unpaidAmount` — Current unpaid balance in ETB

## Related ML Features

- `net_income` — Total income minus total cost; outstanding loan repayments are a cost that reduces net income
- `total_estimated_cost` — Aggregate cost feature that may incorporate loan servicing obligations

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Outstanding debt is a fundamental credit risk indicator across all lending contexts. In Ethiopian smallholder agriculture, existing loans are particularly concerning because:

1. **Over-indebtedness risk**: Farmers who already owe money to one lender and take on additional debt face compounding repayment obligations that may exceed their total income. The 10,000 ETB threshold reflects a meaningful debt level for smallholder farmers — above this amount, the combined debt service becomes difficult to sustain on typical farm incomes.

2. **Multiple borrowing**: Many smallholder farmers borrow from both formal (microfinance, banks) and informal (family, local moneylenders, EQUIB) sources simultaneously. The survey captures both formal and informal credit access to provide a complete picture of the farmer's debt position.

3. **Repayment history signal**: The presence of outstanding debt, combined with Q38 (repayment difficulty), indicates whether the farmer has a pattern of struggling with debt obligations. A farmer who currently owes money AND has experienced repayment difficulty is a substantially higher risk than one with a clean repayment history.

This data point has the lowest reliability rating (0.25) in the framework because outstanding loan amounts are entirely self-reported and farmers have a strong incentive to understate their existing debt when applying for new credit. Informal loans from family or community members are particularly likely to be omitted. The low reliability weight ensures that even if a farmer understates their debt, the scoring impact is limited — the system relies more heavily on objective, verifiable data points (like DP5 and DP6) for the primary creditworthiness assessment.
