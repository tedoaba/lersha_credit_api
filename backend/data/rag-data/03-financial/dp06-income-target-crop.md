# DP6: Income from Target Crop Sales

## Summary

Income from Target Crop Sales measures the projected revenue the farmer will generate from selling their primary crop harvest. Like DP5 (cost), this data point uses externally validated crop reference data for yield-per-hectare and market prices, combined with the farmer's reported marketable yield percentage and cultivation area. Higher projected revenue indicates stronger repayment capacity. This is one of the two highest-weighted data points in the entire scoring framework.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: High (1.0)
- **Framework Design Weight**: 15.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 15 points

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

Revenue is derived from the CropRef external data table:

```
Target_Crop_Revenue = XLOOKUP(Target_Crop, CropRef!Crop_Column, CropRef!Revenue_Per_Ha) × Target_Crop_Area_Ha
```

**Revenue per hectare calculation in CropRef:**
```
Revenue_Per_Ha = Yield_Per_Ha (quintals) × Marketable_Percentage × Price_Per_Quintal (ETB)
```

**Step-by-step:**
1. The farmer selects their target crop (Q25) and reports cultivation area (Q26) and marketable yield percentage (Q27).
2. The system looks up the standardized revenue per hectare from CropRef.
3. Per-hectare revenue is multiplied by the farmer's area.
4. The total revenue is quartile-scored — farmers with higher revenue receive higher scores.

## Scoring Tiers

| Quartile | Revenue Level | Score Direction | Rationale |
|----------|--------------|-----------------|-----------|
| 1st Quartile | Lowest revenue | Lowest score | Limited income generation; weak repayment capacity |
| 2nd Quartile | Below average | Below-average score | Modest revenue; some repayment risk |
| 3rd Quartile | Above average | Above-average score | Strong revenue generation |
| 4th Quartile | Highest revenue | Highest score (15) | Maximum income potential; strongest repayment capacity |

## Reference Revenue Data by Crop (ETB per hectare)

| Crop | Yield/Ha (qtl) | Marketable % | Price/Qtl (ETB) | Revenue/Ha (ETB) | Profit Margin/Ha (ETB) |
|------|---------------|-------------|----------------|-----------------|----------------------|
| Coffee | 35 | 95% | 3,500 | 116,375 | 112,375 |
| Tomato | 150 | 95% | 400 | 57,000 | 52,150 |
| Potatoes | 200 | 85% | 300 | 51,000 | 45,800 |
| Maize | 30 | 85% | 1,600 | 40,800 | 35,785 |
| Onion | 120 | 90% | 350 | 37,800 | 32,520 |
| Malt Barley | 25 | 80% | 1,800 | 36,000 | 32,060 |
| Soybean | 18 | 90% | 2,200 | 35,640 | 33,640 |
| Chickpea | 15 | 85% | 2,400 | 30,600 | 28,760 |
| Sesame | 8 | 90% | 4,000 | 28,800 | 27,240 |
| Wheat | 22 | 75% | 1,700 | 28,050 | 24,660 |
| Haricot Bean | 12 | 90% | 2,600 | 28,080 | 26,575 |
| Sorghum | 18 | 75% | 1,500 | 20,250 | 17,220 |
| Teff | 12 | 70% | 2,000 | 16,800 | 15,040 |
| Millet | 10 | 80% | 1,400 | 11,200 | 9,320 |

## Survey Questions

- **Q25**: What is your target crop for this season? (Dropdown: crop selection)
- **Q26**: What is the farm size in hectares for your target crop? (Numeric, 0–10)
- **Q27**: What percentage of your target crop yield is marketable? (Numeric, 0–100%)

## Database Fields

- `mainCrops` — Primary crop planted
- `FarmSizeHectares` — Land allocated to target crop (hectares)
- `ExpectedYieldQuintals` — Projected yield in quintals
- `SaleableYieldQuintals` — Marketable portion of harvest in quintals
- `LastYearAveragePrice` — Historical reference price per quintal

## Related ML Features

- `expectedyieldquintals` — Expected harvest yield in quintals
- `saleableyieldquintals` — Quantity intended for market sale
- `yield_per_hectare` — Key productivity indicator (expected yield / hectares)
- `total_estimated_income` — Sum of primary farm income and other activities
- `net_income` — Total income minus total cost (primary creditworthiness indicator)

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Revenue from the target crop is the single most direct indicator of a farmer's ability to repay a loan. The income generated from crop sales is the primary source of cash flow for most smallholder farmers, and the size of that income relative to the loan amount determines whether repayment is feasible.

The use of CropRef standardized data is particularly important for revenue projections. Farmers naturally tend to be optimistic about their expected yields and prices, which would inflate self-reported revenue estimates. By using externally validated yield-per-hectare data and current market prices from the reference table, the system produces realistic revenue projections that are consistent across all farmers growing the same crop in similar conditions.

The revenue-to-cost ratio (DP6 / DP5) provides the fundamental profitability assessment. A farmer growing coffee (revenue 116,375 ETB/ha vs. cost 4,000 ETB/ha) has dramatically different repayment capacity than a farmer growing millet (revenue 11,200 ETB/ha vs. cost 1,880 ETB/ha), even if both cultivate the same land area. This data point, combined with DP5, captures that difference with high reliability (1.0) because both values are derived from objective external data rather than farmer estimates.
