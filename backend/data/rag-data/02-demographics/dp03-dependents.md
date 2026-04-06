# DP3: Number of Dependents

## Summary

Number of Dependents measures the household burden relative to the farming household's productive capacity. A larger number of dependents — particularly children under 12 and elderly members over 60 — increases the household's consumption needs without proportionally increasing income-generating capacity. This data point inversely correlates with creditworthiness: fewer dependents relative to household resources means more disposable income available for loan repayment.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: Average (0.5)
- **Framework Design Weight**: 5.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 5 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

The total number of dependents is derived from the household size reported in Q11, combined with the breakdown of children under 12 (Q12) and elderly over 60 (Q13). The raw count is scored using quartile-based ranking against the population distribution:

```
Score = QUARTILE-based scoring of household_size
```

Farmers with fewer dependents (lower quartiles) receive higher scores. The scoring inverts the ranking so that a smaller household burden translates to a higher creditworthiness score.

## Scoring Tiers

| Quartile Position | Relative Household Size | Score Direction | Rationale |
|-------------------|------------------------|-----------------|-----------|
| 1st Quartile | Smallest households | Highest score | Minimal consumption burden; maximum repayment capacity |
| 2nd Quartile | Below average | Above-average score | Manageable dependent load |
| 3rd Quartile | Above average | Below-average score | Elevated consumption requirements |
| 4th Quartile | Largest households | Lowest score | High dependent burden strains cash flow |

## Survey Questions

- **Q11**: Total number of family members in the household (Numeric, range 0–20)
- **Q12**: Number of children under 12 years old (Numeric, range 0–10)
- **Q13**: Number of elderly members over 60 years old (Numeric, range 0–10)

## Database Fields

- `totalFamilyMembers` — Total household size (number, 0–20)
- `childrenUnder12` — Count of children below 12 years (number, 0–10)
- `elderlyMembersOver60` — Count of household members above 60 years (number, 0–10)

## Related ML Features

- `family_size` — Total number of family members, directly used as a model feature. Affects income-per-member calculations and loan capacity assessment.
- `income_per_family_member` — Derived feature: total estimated income divided by family size. This welfare indicator captures the per-capita economic capacity of the household.

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Household size directly impacts a farmer's ability to service debt. In Ethiopian smallholder contexts, larger households face higher food consumption costs, educational expenses (student remittances), and healthcare needs. Children under 12 consume household resources without contributing to farm labor or income, while elderly members over 60 may require medical care and reduce the household's ability to mobilize labor during peak agricultural seasons.

The dependency ratio — the proportion of non-productive members to productive members — is a well-established predictor of household financial stress. A farmer with 2 dependents and strong crop income has fundamentally different repayment capacity than a farmer with 8 dependents and the same crop income. The per-capita income metric (income_per_family_member) derived from this data point provides a more nuanced view of actual household welfare than total income alone.

This data point has Average reliability (0.5) because while household size is generally known, the exact count of family members can fluctuate with seasonal migration, extended family arrangements common in Ethiopian households, and varying definitions of "family member." Field agents verify where possible, but some imprecision is expected.
