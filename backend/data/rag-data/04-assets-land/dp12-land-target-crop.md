# DP12: Land Size for Target Crop

## Summary

Land Size for Target Crop measures the specific area allocated to the farmer's primary crop for the current season. This data point bridges the gap between total land managed (DP10) and the actual production commitment for the loan-financed crop. It directly feeds into the cost (DP5) and revenue (DP6) calculations by providing the area multiplier for per-hectare reference data. Larger allocations to the target crop indicate greater commitment and higher potential revenue from the specific activity being financed.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: Average (0.5)
- **Framework Design Weight**: 5.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 5 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

Target crop land area is reported directly from Q26 and scored using quartile-based ranking:

```
Score = QUARTILE-based scoring of Target_Crop_Area_Ha
```

Farmers allocating more land to their target crop receive higher scores.

## Scoring Tiers

| Quartile | Area Allocated | Score Direction | Rationale |
|----------|---------------|-----------------|-----------|
| 1st Quartile | Smallest allocation | Lowest score | Limited target crop commitment |
| 2nd Quartile | Below average | Below-average score | Modest allocation |
| 3rd Quartile | Above average | Above-average score | Significant commitment to target crop |
| 4th Quartile | Largest allocation | Highest score (5) | Maximum production commitment; highest revenue potential |

## Survey Questions

- **Q25**: What is your target crop for this season? (Dropdown selection)
- **Q26**: What is the farm size in hectares allocated to your target crop? (Numeric, range 0–10)

## Database Fields

- `FarmSizeHectares` — Land area allocated to the target crop in hectares
- `mainCrops` — The selected target crop

## Related ML Features

- `farmsizehectares` — Total operated farm area (includes target crop allocation)
- `yield_per_hectare` — Expected yield per hectare, calculated using target crop area as the denominator
- `input_intensity` — Input-to-land ratio: (seeds + urea + DAP) / farmsize

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

The land area allocated to the target crop is the key multiplier in both cost and revenue calculations. When a farmer allocates 2 hectares to wheat versus 0.5 hectares, their projected revenue (and cost) scales proportionally — the CropRef per-hectare values are multiplied by this area to produce the farmer's specific financial projections.

This data point also reveals the farmer's crop diversification strategy. A farmer who allocates all their land to a single target crop is concentrating risk, while one who allocates a portion to the target crop and diversifies with other crops (captured in DP7 and DP8) is spreading risk. Both strategies have merit — concentration maximizes revenue from the most profitable crop, while diversification provides insurance against crop-specific failures.

The relationship between DP12 (target crop area) and DP10 (total land managed) is informative. If a farmer manages 5 hectares total but allocates only 1 hectare to their target crop, the remaining 4 hectares may be used for subsistence crops, livestock grazing, or left fallow. This ratio indicates how much of their productive capacity is directed toward the income-generating activity that will fund loan repayment.

This data point has Average reliability (0.5) because while the area allocation is a current-season planning decision that the farmer knows, precise hectare estimates can be challenging for smallholders who may not have surveyed plots.
