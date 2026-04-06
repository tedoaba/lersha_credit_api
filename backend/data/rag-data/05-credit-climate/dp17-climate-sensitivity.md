# DP17: Sensitivity of Target Crop to Climate Variability

## Summary

Sensitivity of Target Crop to Climate Variability measures how vulnerable the farmer's primary crop is to rainfall changes, temperature stress, and climate-related yield impacts. Some crops are inherently more resilient to climate variability than others — drought-tolerant sorghum, for example, is less sensitive than water-dependent tomatoes. Farmers growing less climate-sensitive crops face lower production risk and, therefore, lower loan default risk. Lower sensitivity results in a higher score.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: Low (0.25)
- **Framework Design Weight**: 3.75
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 3.75 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP17 averages multiple sensitivity indicators and applies inverse quartile scoring:

```
Sensitivity_Index = Average(Rainfall_Variability_Impact, Crop_Rainfall_Sensitivity, Climate_Income_Impact)
Score = Inverse quartile ranking of Sensitivity_Index
```

Farmers reporting lower crop sensitivity to climate receive higher scores.

## Sensitivity Scale

| Response | Sensitivity Level | Implication |
|----------|------------------|-------------|
| Not at all / Not sensitive | Minimal climate impact | Highly resilient crop/system |
| Slightly | Minor yield variation | Low production risk |
| Moderately | Noticeable yield changes | Moderate production risk |
| Very | Significant yield reduction | High production risk |
| Extremely | Severe yield loss or crop failure | Critical production risk |

## Crop Sensitivity Profiles

| Crop | Drought Sensitivity | Flood Sensitivity | Overall Climate Risk |
|------|-------------------|-------------------|---------------------|
| Teff | Moderate | Low | Moderate — relatively adapted to Ethiopian highlands |
| Sorghum | Low | Low | Low — drought-tolerant, widely adapted |
| Millet | Low | Low | Low — highly drought-resistant |
| Maize | High | Moderate | High — sensitive to moisture stress at flowering |
| Wheat | Moderate | Moderate | Moderate — needs reliable rainfall timing |
| Malt Barley | Moderate | Moderate | Moderate — highland-adapted but timing-dependent |
| Tomato | Very High | High | Very High — water-intensive, pest-vulnerable |
| Onion | High | High | High — sensitive to both drought and excess moisture |
| Potatoes | High | High | High — requires consistent moisture and cool temperatures |
| Soybean | Moderate | Moderate | Moderate — nitrogen-fixing but moisture-sensitive |
| Coffee | Moderate | Low | Moderate — perennial with deep roots, but narrowly adapted |
| Sesame | Low | Moderate | Low-Moderate — heat-tolerant, drought-adapted |

## Survey Questions

- **Q43.4**: How does rainfall variability impact your crop yield? (Not at all / Slightly / Moderately / Very / Extremely)
- **Q43.5**: How sensitive is your target crop to rainfall timing? (Not sensitive / Slightly / Moderately / Very / Extremely sensitive)
- **Q43.6**: How has climate variability impacted your income from farming? (Not at all / Slightly / Moderately / Very / Extremely)

## Database Fields

Climate sensitivity data is captured through the survey's climate profile section.

## Related ML Features

- `yield_per_hectare` — Climate sensitivity directly impacts achievable yields; more sensitive crops show higher yield variance
- `expectedyieldquintals` — Expected yield reflects the farmer's own assessment of likely production given climate conditions

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Crop sensitivity to climate is a structural risk factor that differs from farmer-to-farmer based on their crop choice and local agro-ecology. A farmer growing tomatoes in a drought-prone area faces compounding risk — high exposure (DP16) multiplied by high sensitivity (DP17) — that creates a very challenging repayment environment.

The interaction between DP16 and DP17 is particularly informative:
- **Low exposure + Low sensitivity** = Minimal climate risk (e.g., sorghum in a stable rainfall zone)
- **Low exposure + High sensitivity** = Manageable risk if conditions remain stable (e.g., tomatoes with reliable irrigation)
- **High exposure + Low sensitivity** = Moderate risk mitigated by crop resilience (e.g., millet in drought-prone areas)
- **High exposure + High sensitivity** = Maximum climate risk (e.g., maize in erratic rainfall zones)

This data point helps the credit scoring system account for crop-specific risk that cannot be captured by financial metrics alone. Two farmers with identical income projections may have vastly different risk profiles if one grows drought-tolerant sorghum and the other grows water-dependent tomatoes.

Like DP16, this data point has Low reliability (0.25) because sensitivity assessments are based on farmer perceptions rather than agronomic measurements. Different farmers growing the same crop may report different sensitivity levels based on their personal experience, micro-climate conditions, and management practices.
