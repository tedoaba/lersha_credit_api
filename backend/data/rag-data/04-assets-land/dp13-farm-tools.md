# DP13: Farm Tool Ownership and Control

## Summary

Farm Tool Ownership and Control evaluates the type of agricultural implements the farmer uses and, critically, who controls those tools. Modern or mechanized farm tools indicate higher productivity potential and capital investment, while control over tools (as opposed to dependence on a spouse or other party) indicates operational autonomy. This data point combines an asset assessment with a gender-intentional agency measure.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: Average (0.5)
- **Framework Design Weight**: 5.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 5 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP13 is scored based on two factors: the type of tools and who controls them.

```
Tool_Score = IF(Modern_Mechanized, higher_score, IF(Traditional, lower_score, 0))
Control_Bonus = IF(Tool_Controller = "Me", bonus, 0)
DP13 = Tool_Score + Control_Bonus
```

The combined score is then quartile-ranked against the population.

## Scoring Components

| Component | Condition | Score Impact | Rationale |
|-----------|-----------|-------------|-----------|
| Tool type | Modern/mechanized | Higher score | Greater productivity, capital investment |
| Tool type | Traditional | Lower score | Limited productivity, minimal investment |
| Tool control | "Me" (self) | Bonus points | Operational autonomy |
| Tool control | "Spouse" or "Other" | No bonus | Dependent on others for productive operations |

## Survey Questions

- **Q24**: What type of farm tools do you use? (Traditional / Modern)
  - Sub-question: Who controls the farm tools? (Me / Spouse / Other) — gender-intentional
- **Q28** (supplementary): Do you have access to farm mechanization services? (Yes/No)

## Database Fields

- `farm_tools` — Type of tools: Traditional vs Modern
- `Farm_Mechanization` — Access to mechanization services (boolean)
- `decision_making_role` — Role: plot owner, manager, or labor (contextual)

## Related ML Features

- `farm_mechanization` — Categorical feature: manual, semi-mechanized, fully mechanized. Indicates the level of technology adoption in farming operations.
- `asset_ownership` — Binary feature for productive asset ownership (1=yes, 0=no). Farm tools are a key component of productive assets.

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Farm tool type is a direct indicator of agricultural productivity potential. Farmers using modern or mechanized tools can prepare land faster, plant more efficiently, manage crops better, and harvest with less loss. This productivity advantage translates to higher yields per hectare and, consequently, stronger income generation and loan repayment capacity.

The control dimension is equally important from a gender-intentional perspective. In many Ethiopian households, farm tools may be owned by the household but controlled by one member. A woman farmer who controls her own tools can make independent decisions about when and how to farm — she does not need to negotiate access or timing with a spouse or relative. This operational autonomy directly impacts her ability to optimize farm operations and, by extension, her income and repayment capacity.

Access to farm mechanization services (tractors, threshers, irrigation pumps) represents a further productivity multiplier. Even farmers who do not own mechanized equipment can benefit from rental or cooperative access to these services. The combination of tool ownership and mechanization access provides a comprehensive picture of the farmer's technological capacity for productive agriculture.

This data point has Average reliability (0.5) because tool type is observable during field visits, but the distinction between "traditional" and "modern" can be subjective, and the control question relies on self-reporting that may not fully capture intra-household dynamics.
