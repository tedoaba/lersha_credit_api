# DP5: Cost of Planned Target Crop Farming

## Summary

Cost of Target Crop Farming measures the total input cost required to cultivate the farmer's primary crop for the season. This data point uses externally validated crop reference data (CropRef) rather than farmer self-reports, making it one of the most reliable financial indicators in the scoring framework. Lower production costs relative to expected revenue indicate a more favorable cost structure and greater repayment capacity. The cost includes seeds, fertilizers (Urea, DAP/NPS), and crop protection chemicals, all standardized per hectare and scaled by the farmer's target crop area.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: High (1.0)
- **Framework Design Weight**: 15.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 15 points

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

The cost is derived from the CropRef external data table using a lookup formula:

```
Target_Crop_Cost = XLOOKUP(Target_Crop, CropRef!Crop_Column, CropRef!Total_Cost_Per_Ha) × Target_Crop_Area_Ha
```

**Step-by-step:**
1. The farmer selects their target crop (Q25) and reports their planned cultivation area in hectares (Q26).
2. The system looks up the standardized total cost per hectare for that crop from the CropRef table.
3. The per-hectare cost is multiplied by the farmer's area to get the total projected cost.
4. The total cost is then quartile-scored against the population distribution — farmers with lower costs receive higher scores.

The CropRef total cost per hectare includes:
- Seed cost: quantity per hectare × price per unit
- Urea fertilizer: quantity per hectare × price per quintal
- DAP/NPS fertilizer: quantity per hectare × price per quintal
- Crop protection chemicals: up to 3 chemicals × quantity × price

## Scoring Tiers

| Quartile | Cost Level | Score Direction | Rationale |
|----------|-----------|-----------------|-----------|
| 1st Quartile | Lowest costs | Highest score (15) | Minimal financial exposure; strong margin potential |
| 2nd Quartile | Below average | Above-average score | Manageable input requirements |
| 3rd Quartile | Above average | Below-average score | Higher financial commitment required |
| 4th Quartile | Highest costs | Lowest score | Significant capital outlay increases default risk |

## Reference Cost Data by Crop (ETB per hectare)

| Crop | Total Cost/Ha (ETB) | Key Cost Drivers |
|------|---------------------|------------------|
| Maize | 5,015 | High fertilizer needs (Urea 200kg + DAP 120kg) |
| Wheat | 3,390 | Moderate inputs, fungicide protection |
| Malt Barley | 3,940 | Balanced seed and fertilizer costs |
| Teff | 1,760 | Low input requirements (minimal fertilizer) |
| Sorghum | 3,030 | Moderate inputs with herbicide |
| Millet | 1,880 | Low seed and chemical costs |
| Soybean | 2,000 | No Urea needed (nitrogen-fixing legume) |
| Haricot Bean | 1,505 | Lowest cost crop; no Urea needed |
| Chickpea | 1,840 | Low input cost; no Urea needed |
| Tomato | 4,850 | High seedling + Urea costs |
| Onion | 5,280 | High across all input categories |
| Potatoes | 5,200 | High seed tuber + fertilizer costs |
| Sesame | 1,560 | Minimal inputs; herbicide only |
| Coffee | 4,000 | Seedling establishment cost; no fertilizer |

## Survey Questions

- **Q25**: What is your target crop for this season? (Dropdown: Malt Barley / Wheat / Maize / Soybean / Tomato / Onion / Potatoes / Other)
- **Q26**: What is the farm size in hectares allocated to your target crop? (Numeric, range 0–10)

## Database Fields

- `mainCrops` — Primary crop planted for the season
- `FarmSizeHectares` — Land area allocated to the target crop in hectares

## Related ML Features

- `farmsizehectares` — Total operated farm area, used to scale cost calculations
- `seedquintals` — Seed quantity used in quintals
- `ureafertilizerquintals` — Urea fertilizer quantity in quintals
- `dapnpsfertilizerquintals` — DAP/NPS fertilizer quantity in quintals
- `input_intensity` — Derived ratio: (seeds + urea + DAP) / farmsize, indicating farming intensity
- `total_estimated_cost` — Sum of production expenses and estimated total costs

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Production cost is one of the two most important financial predictors (alongside revenue, DP6) in the credit scoring framework, carrying a maximum weighted score of 15 points — the highest possible weight. The cost of farming directly determines how much capital a farmer needs to invest before generating any return, and thus how much financial pressure they face during the growing season.

Farmers cultivating lower-cost crops (like teff, sesame, or legumes) face less financial risk because their upfront investment is smaller. If the harvest is poor, their losses are more manageable. In contrast, high-cost crops like onions, potatoes, and tomatoes require significant capital outlays that, if the season fails, create debt obligations the farmer cannot service.

Crucially, DP5 uses externally validated CropRef data rather than farmer self-reports. This eliminates the possibility of farmers understating their costs to appear more creditworthy. The CropRef table is maintained with standardized input quantities and current market prices, ensuring that cost projections are realistic and comparable across farmers growing the same crop. This objectivity is why DP5 has a reliability rating of 1.0 (High) — the highest possible.
