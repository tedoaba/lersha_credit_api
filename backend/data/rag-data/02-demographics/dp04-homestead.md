# DP4: Family Homestead Type

## Summary

Family Homestead Type uses the farmer's dwelling as a proxy for accumulated wealth and asset stability. The type of house a farmer lives in reflects years of investment and financial capacity — modern housing indicates significant capital accumulation, while traditional all-in-one dwellings suggest more limited resources. This data point serves as an observable, verifiable indicator of long-term financial standing that complements income-based measures.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: Average (0.5)
- **Framework Design Weight**: 5.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 5 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

Homestead type is scored using a direct tier mapping based on the house type reported in Q14:

```
IF(House_Type = "Modern", 5, IF(House_Type = "Traditional with multiple rooms", 2.5, 0))
```

## Scoring Tiers

| House Type | Score | Rationale |
|-----------|-------|-----------|
| Modern | 5.0 | Significant capital investment; indicates strong asset accumulation and financial stability |
| Traditional with multiple rooms | 2.5 | Moderate investment; demonstrates some wealth accumulation beyond subsistence |
| Traditional all-in-one | 0.0 | Minimal housing investment; suggests limited surplus capital |

## Survey Questions

- **Q14**: What is the type of your family homestead? (Dropdown: Traditional all-in-one / Traditional with multiple rooms / Modern)

### House Type Definitions

- **Traditional all-in-one**: A single-room dwelling typically constructed with mud walls, thatched roof, and earthen floor. The household uses one room for sleeping, cooking, and storage. Common in rural areas with limited infrastructure.
- **Traditional with multiple rooms**: A dwelling with separate rooms for different functions (sleeping, cooking, storage) but built with traditional materials. Indicates investment in improving living conditions beyond the minimum.
- **Modern**: A dwelling constructed with permanent materials such as corrugated iron roofing, concrete or brick walls, and cement flooring. Often includes multiple rooms, separate kitchen, and possibly improved water/sanitation. Represents substantial capital investment.

## Database Fields

- `typeOfHouse` — Housing type classification (Traditional/Modern)

## Related ML Features

- `typeofhouse` — Categorical feature: permanent, semi-permanent, or temporary. Maps from the three survey categories to the ML model's three-level classification.

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Housing type is one of the most reliable asset proxies available in smallholder agricultural contexts where formal financial records are scarce. A modern house in rural Ethiopia represents an investment of tens of thousands of ETB accumulated over years, demonstrating the farmer's ability to generate surplus income beyond subsistence needs. This surplus-generation capacity directly predicts loan repayment ability.

The progression from traditional all-in-one to multiple rooms to modern housing follows a well-documented wealth accumulation pathway in Ethiopian rural communities. Farmers who have reached the modern housing tier have typically done so through sustained agricultural productivity, diversified income, or successful prior credit relationships. Their housing investment also serves as an informal collateral indicator — farmers with more invested in their homestead have more to lose from defaulting on credit obligations.

This data point has Average reliability (0.5) because while house type is directly observable by the field agent during the survey visit, the three-tier classification is somewhat subjective. The boundary between "traditional with multiple rooms" and "modern" can vary by regional standards, and some farmers may have mixed construction (e.g., corrugated roof with mud walls). Despite this, the general wealth signal remains valid across classifications.
