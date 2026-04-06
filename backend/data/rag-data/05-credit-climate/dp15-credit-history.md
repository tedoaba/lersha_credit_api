# DP15: Credit History

## Summary

Credit History captures the farmer's past experience with formal and informal credit, including whether they have accessed credit, their repayment track record, savings behavior, and participation in contract farming. Past repayment behavior is widely recognized as the strongest single predictor of future repayment behavior. However, in the Ethiopian smallholder context, credit history data is largely self-reported and often incomplete due to the prevalence of informal lending, resulting in a low reliability rating.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: Low (0.25)
- **Framework Design Weight**: 3.75
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 3.75 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP15 is a composite score based on multiple credit behavior indicators:

```
Credit_Score = Count of positive credit behaviors from:
  - Has_Savings_Account (Q41): +points
  - Contract_Farming (Q42): +points
  - No_Repayment_Difficulty (Q38 = No): +points
  - No_Outstanding_Default: +points
  - Formal_Credit_Access (Q37 = Yes): +points
```

The count-based score is then quartile-ranked against the population.

## Credit Behavior Components

| Indicator | Source | Positive Signal | Negative Signal |
|-----------|--------|-----------------|-----------------|
| Informal credit access | Q36 | Community trust and social capital | May indicate financial stress |
| Formal credit in last 12 months | Q37 | Institutional engagement | Higher debt exposure |
| Repayment difficulty | Q38 | No difficulty = strong track record | Difficulty = elevated risk |
| Outstanding loan | Q39–Q40 | No loan = clean slate | Large balance = burden |
| Savings account | Q41 | Financial discipline and planning | Absence = limited financial management |
| Contract farming | Q42 | Market linkage and commitment | Absence = less structured operations |

## Survey Questions

- **Q36**: Do you use informal credit sources (family, friends, local lenders)? (Yes/No)
- **Q37**: Have you had access to formal credit in the last 12 months? (Yes/No)
- **Q38**: Have you experienced difficulty repaying any loan? (Yes/No)
- **Q39**: Do you currently have an outstanding loan? (Yes/No)
- **Q40**: Total outstanding loan amount (0–100,000 ETB)
- **Q41**: Do you have a savings account with a bank, MFI, or RUSACCO? (Yes/No)
- **Q42**: Do you have contract farming experience? (Yes/No)

## Database Fields

- `hasAccessToCreditSource` — Credit access in last 12 months (boolean)
- `receivedFromMicrofinance` — Microfinance/bank credit receipt (boolean)
- `microfinanceName` — Name of credit institution (string)
- `amountBorrowed` — Loan amount in ETB
- `unableToRepayCreditOnTime` — Repayment difficulty history (boolean)
- `unpaidAmount` — Current unpaid balance in ETB
- `HasSavingAccount` — Formal savings account existence (boolean)

## Related ML Features

- `net_income` — Primary creditworthiness indicator; credit history provides context for interpreting net income as a repayment predictor
- `institutional_support_score` — Sum of institutional flags including microfinance membership

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Credit history is paradoxically the most predictive yet least reliable data point in the framework. In formal banking, credit history (via credit bureaus) is the foundation of credit scoring. In Ethiopian smallholder agriculture, however, most farmers have limited formal credit history because they have been excluded from the formal financial system.

**Savings account ownership** (Q41) is one of the most informative components. A farmer who maintains a savings account with a bank, MFI, or RUSACCO demonstrates financial planning ability, discipline in setting aside money, and engagement with formal financial institutions. These behaviors strongly predict responsible loan management.

**Contract farming experience** (Q42) indicates that the farmer has worked within structured agricultural agreements with buyers, which requires meeting quality standards, delivery schedules, and sometimes input repayment obligations. This experience closely mirrors the discipline needed for loan repayment.

**Repayment difficulty** (Q38) is the most direct risk indicator. A farmer who has previously struggled to repay loans faces a significantly higher probability of future default, especially if the underlying conditions (land size, crop choice, household expenses) have not changed.

The low reliability rating (0.25) reflects the fundamental challenge of self-reported credit data. Farmers seeking credit have incentives to overstate their savings, understate their debts, and omit past repayment difficulties. Informal loans from family and community are particularly likely to be unreported. The scoring framework addresses this by giving credit history a high predictive power (15) but heavily discounting it through the low reliability weight, yielding an effective maximum of only 3.75 points.
