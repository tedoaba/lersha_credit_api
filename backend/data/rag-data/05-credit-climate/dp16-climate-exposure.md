# DP16: Exposure to Extreme Climate Events

## Summary

Exposure to Extreme Climate Events measures how frequently the farmer experiences climate-related shocks such as droughts, floods, and pest/disease outbreaks. In rain-fed Ethiopian agriculture, climate events are a primary driver of crop failure and, consequently, loan default. Farmers in areas with frequent extreme events face higher repayment risk regardless of their financial capacity, because a single climate shock can eliminate an entire season's income. Lower exposure frequency results in a higher score.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: Low (0.25)
- **Framework Design Weight**: 3.75
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 3.75 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP16 is derived from an average of three climate frequency indicators, then quartile-scored inversely (lower exposure = higher score):

```
Exposure_Index = Average(Drought_Frequency, Flood_Frequency, Pest_Disease_Frequency)
Score = Inverse quartile ranking of Exposure_Index
```

Farmers reporting less frequent extreme events receive higher scores.

## Frequency Scale

The survey uses a standardized frequency scale for each climate event type:

| Response | Frequency Level | Implication |
|----------|----------------|-------------|
| Never | No events in recent memory | Lowest risk zone |
| Rarely | Once in several years | Low risk |
| Sometimes | Every few seasons | Moderate risk |
| Frequently | Most seasons affected | High risk |
| Very Frequently | Every season affected | Highest risk; chronic climate stress |

## Climate Event Types

| Event Type | Description | Agricultural Impact |
|-----------|-------------|-------------------|
| Drought | Extended dry periods, rainfall deficit | Crop wilting, yield reduction, total crop failure |
| Flooding | Excess rainfall, waterlogging | Crop damage, soil erosion, root rot, infrastructure damage |
| Pest/disease outbreaks | Insect infestations, fungal or bacterial diseases | Partial to complete yield loss, increased chemical costs |

## Survey Questions

- **Q43.3**: How has the frequency of pest and disease outbreaks changed? (Never / Rarely / Sometimes / Frequently / Very Frequently)
- **Q43.4**: How does rainfall variability impact your crop yield? (Not at all / Slightly / Moderately / Very / Extremely) — feeds into both DP16 and DP17

## Database Fields

Climate exposure data is captured through the survey's climate profile section and stored as part of the farmer's assessment record.

## Related ML Features

- `flaw` — Presence of observed land/crop defects or quality issues (1=yes, 0=no). May indicate current or recent climate damage.

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Climate exposure is fundamentally different from other credit risk factors because it represents an external, uncontrollable hazard. A farmer can improve their management practices, invest in better inputs, or diversify income — but they cannot prevent a drought or flood. In Ethiopia, where approximately 95% of cropland is rain-fed, climate variability is the single largest source of agricultural production risk.

The scoring logic inversely weights exposure — farmers in lower-exposure areas receive higher scores. This reflects the reality that a farmer in a drought-prone zone of the Rift Valley faces structurally higher default risk than an identically capable farmer in a more climatically stable highland area, regardless of their farming skill or financial management.

The combination of DP16 (exposure), DP17 (sensitivity), and DP18-20 (adaptive capacity) creates a comprehensive climate risk profile aligned with established vulnerability assessment frameworks. A farmer may be highly exposed to drought (low DP16 score) but grow drought-resistant crops (high DP17 score) and have irrigation access (high DP18 score), resulting in a moderate overall climate risk profile.

This data point has Low reliability (0.25) because climate frequency is based on farmer perception and recall, which can be imprecise. Farmers may conflate frequency with severity, misremember the timing of events, or be influenced by recent experiences (recency bias). The low reliability weight ensures that climate exposure contributes context to the scoring without dominating the overall assessment.
