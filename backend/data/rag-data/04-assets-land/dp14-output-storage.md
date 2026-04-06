# DP14: Output Storage and Control

## Summary

Output Storage and Control evaluates where the farmer stores their harvested crops and who controls the decision to sell. Post-harvest storage is critical in agricultural economics — farmers who can store their harvest safely and sell at optimal market times achieve significantly higher prices than those forced to sell immediately at harvest when prices are typically lowest. Control over storage and sales decisions is a gender-intentional indicator of economic agency.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: Low (0.25)
- **Framework Design Weight**: 2.5
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 2.5 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP14 is scored based on storage type and control:

```
Storage_Score = based on storage type (commodity financing > aggregator > own storage > none)
Control_Bonus = IF(Output_Controller = "Me", bonus, 0)
DP14 = Storage_Score + Control_Bonus
```

## Scoring Components

| Component | Condition | Score Impact | Rationale |
|-----------|-----------|-------------|-----------|
| Storage type | Commodity financing/warehouse receipt | Highest | Formal storage with market access; can use stored grain as collateral |
| Storage type | Aggregator/cooperative storage | Medium-high | Organized collective storage with better market linkage |
| Storage type | Own storage facility | Medium | Independent storage but may lack optimal conditions |
| Storage type | No dedicated storage | Lowest | Forced to sell at harvest; loses price timing advantage |
| Output sale control | "Me" (self) | Bonus | Autonomy over when and at what price to sell |
| Output sale control | "Spouse" or "Other" | No bonus | Dependent on others for sales decisions |

## Survey Questions

- **Q29**: Where do you store your farm output? (Own storage / Aggregator / Commodity financing warehouse)
  - Sub-question: Who controls the sale of farm output? (Me / Spouse / Other) — gender-intentional

## Database Fields

- `storageType` — Storage method for produced output (with control indicator)

## Related ML Features

- `output_storage_type` — Categorical feature: warehouse, silo, none. Captures the farmer's post-harvest management capability.

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Post-harvest loss and price timing are among the most significant determinants of actual farm income in smallholder agriculture. In Ethiopia, crop prices typically drop 20–40% immediately after harvest due to market flooding, then recover over the following months as supply decreases. Farmers who can store their harvest and sell gradually at higher prices can earn significantly more than those who must sell everything at the farmgate during harvest.

**Commodity financing** represents the most sophisticated storage arrangement — the farmer deposits their harvest in a certified warehouse and receives a receipt that can be used as collateral for short-term credit. This enables the farmer to access cash for immediate needs while retaining ownership of the stored crop for sale at optimal prices. This arrangement demonstrates financial sophistication and market linkage.

**Aggregator/cooperative storage** provides similar benefits through collective organization. Farmers who participate in aggregation can negotiate better prices through bulk sales and have access to storage facilities that individual smallholders could not afford.

**Own storage** provides basic protection but may not have optimal conditions (temperature, humidity, pest control), leading to some post-harvest losses. However, it still gives the farmer timing control over sales.

The control dimension measures who decides when to sell and at what price. This is a critical gender equity indicator — in some households, women produce the crop but male family members control the sales proceeds. A woman who controls her own output sales can direct revenue toward loan repayment and household needs she prioritizes.

This data point has Low reliability (0.25) because storage arrangements and control dynamics are difficult to verify independently. Farmers may overstate their storage quality or misrepresent intra-household control dynamics. The low reliability weighting limits the impact of potential misreporting.
