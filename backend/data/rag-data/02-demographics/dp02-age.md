# DP2: Age

## Summary

Age captures the farmer's life stage as a proxy for agricultural experience, physical capacity, and financial stability. The scoring model assigns higher points to farmers in their peak productive and experience years (47–59), moderate points to mid-career farmers (35–46), and lower points to early-career farmers (25–34). Very young farmers (under 25) and elderly farmers (over 59) receive zero points, reflecting limited experience or declining productive capacity respectively.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: High (1.0)
- **Framework Design Weight**: 10.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 10 points

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

Age is scored using a tiered bracket system based on the farmer's reported age:

```
IF(Age < 25, 0, IF(Age <= 34, 2.5, IF(Age <= 46, 5, IF(Age <= 59, 10, 0))))
```

This formula creates four distinct tiers with clear boundaries.

## Scoring Tiers

| Age Range | Score | Rationale |
|-----------|-------|-----------|
| Under 25 | 0.0 | Limited farming experience and unestablished financial track record |
| 25–34 | 2.5 | Early career; developing experience but lower asset accumulation |
| 35–46 | 5.0 | Mid-career; established operations with growing experience |
| 47–59 | 10.0 | Peak experience; maximum institutional knowledge, proven operations, and accumulated assets |
| 60 and above | 0.0 | Declining physical capacity; potential succession uncertainty |

## Survey Questions

- **Q10**: What is your age? (Numeric input, range 18–99)

## Database Fields

- `age` — Farmer's age in years (numeric, derived from Q10)

## Related ML Features

- `age_group` — Derived categorical feature: Young (0–20), Early_Middle (21–35), Late_Middle (36–45), Senior (46+). Note that the ML feature age groups differ slightly from the DP2 scoring brackets.

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Age serves as a reliable proxy for multiple creditworthiness factors in smallholder agriculture. Farmers in the 47–59 age bracket have typically accumulated decades of agricultural knowledge, built social networks, established relationships with input suppliers and buyers, and developed strategies for managing seasonal variability. Their farms tend to be more productive and their operations more stable, translating directly to stronger repayment capacity.

Younger farmers (under 25) often lack the experience base to manage the complexities of rain-fed agriculture effectively, and they may not yet have established the asset base (land, tools, livestock) that provides a financial buffer during difficult seasons. While they may have physical vigor, their higher default risk reflects the learning curve inherent in agricultural management.

Farmers over 60 present a different risk profile: while they possess deep experience, declining physical capacity and potential farm succession transitions can disrupt operations. In Ethiopian smallholder contexts, elderly farmers may transfer management responsibilities to children, creating uncertainty about who will manage loan repayment.

This data point has High reliability (1.0) because age is objectively verifiable through national ID or kebele records, making it resistant to misreporting.
