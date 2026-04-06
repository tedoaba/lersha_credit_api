# DP11: Land Title Availability

## Summary

Land Title Availability indicates whether the farmer holds a formal land title certificate for their owned land. In Ethiopian agricultural lending, land title serves as a critical collateral indicator — it provides legal documentation of the farmer's asset ownership and tenure security. Farmers with formal titles can offer more reliable collateral, reducing lender risk. The scoring also recognizes joint land titles with spouses, supporting gender-intentional credit access.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: High (1.0)
- **Framework Design Weight**: 10.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 10 points

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

Land title is scored as a binary indicator:

```
IF(Has_Land_Title = "Yes", 10, 0)
```

Joint titles with a spouse are counted as "Yes" — the farmer has documented land rights.

## Scoring Tiers

| Condition | Score | Rationale |
|-----------|-------|-----------|
| Has land title (including joint with spouse) | 10.0 | Formal documentation of land rights; strong collateral |
| No land title | 0.0 | No formal land documentation; limited collateral |

## Survey Questions

- **Q23**: Do you have a land title certificate? (Dropdown: Yes / No / Joint with spouse)
  - Note: The "Joint with spouse" option is a gender-intentional addition recognizing women's land rights through joint titling programs.

## Database Fields

- `has_land_title_ownLand` — Whether the farmer has a land title for owned land (boolean)
- `hasParcel_number` — Whether the land has a registered parcel number (boolean)
- `parcel_number` — The registered parcel ID, if available

## Related ML Features

- `land_title` — Binary feature: holds a formal land title document (1=yes, 0=no). Directly used in the ML model as a strong indicator of asset security.

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Land title is one of the most important binary indicators in the credit scoring framework, carrying a full weighted score of 10 points — among the highest for any single data point. Its importance stems from several factors:

**Collateral value**: Formal land title provides lenders with documented security. While Ethiopian land policy generally prohibits land sale (land is state-owned with use rights), a title certificate demonstrates recognized tenure that supports longer-term lending relationships and larger loan sizes.

**Tenure security**: Farmers with formal titles face lower risk of land disputes or involuntary loss of their productive base. This stability means they can plan multi-season investments (including loan-financed inputs) with confidence that they will retain access to the land needed to generate repayment income.

**Financial inclusion signal**: The process of obtaining a land title in Ethiopia requires engagement with formal institutions — kebele administration, land registration offices, and sometimes courts. Farmers who have navigated this process demonstrate institutional engagement skills that correlate with managing formal credit relationships.

**Gender equity**: The joint title option (Q23) is particularly significant for women's credit access. Ethiopia's land certification programs have increasingly included joint titling, where both spouses are named on the land certificate. By recognizing joint titles as equivalent to individual titles for scoring purposes, the system ensures that women benefiting from joint certification programs receive full credit for their documented land rights.

This data point has High reliability (1.0) because land title status is verifiable through physical documentation — the farmer can present their land certificate, and the field agent can confirm its existence. This makes it one of the most objective data points in the framework.
