# DP1: Gender Agency and Control

## Summary

Gender Agency and Control measures the farmer's actual economic decision-making power, not simply their biological gender. This data point captures whether the farmer is the primary decision-maker in the household, controls farm tools, controls the sale of farm output, and accounts for marital status as an indicator of independent economic management. It is a cornerstone of the gender-intentional credit scoring design, ensuring that women who actively manage farm operations receive credit scores that reflect their true capacity.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: High (1.0)
- **Framework Design Weight**: 10.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 10 points

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP1 is a composite score built from five survey indicators:

1. **Gender** (Q9): Female = 5 points, Male = 0 points
2. **Primary Decision-Maker** (Q9.1): Yes = 2 points, No = 0 points
3. **Farm Tool Control** (Q24): Controller is "Me" = 1 point, otherwise = 0
4. **Output Sale Control** (Q29): Controller is "Me" = 1 point, otherwise = 0
5. **Marital Status** (Q9.2): Widowed or Divorced = 1 point, otherwise = 0

The formula sums these components:

```
DP1 = IF(Female, 5, 0) + IF(Primary_Decision_Maker, 2, 0) + IF(Tool_Control="Me", 1, 0) + IF(Output_Control="Me", 1, 0) + IF(Widowed_or_Divorced, 1, 0)
```

A female farmer who is the primary decision-maker, controls her own tools and output sales, and is widowed or divorced would score the maximum of 10 points. A male farmer who controls tools and output but is not classified as primary decision-maker in the gender-intentional framework would score 2 points.

## Scoring Tiers

| Component | Condition | Points | Rationale |
|-----------|-----------|--------|-----------|
| Gender | Female | 5 | Women face systemic barriers to credit access; positive weighting corrects historical bias |
| Gender | Male | 0 | Men have existing advantages in formal credit systems |
| Decision-Maker | Primary decision-maker = Yes | 2 | Active control over household finances predicts responsible loan management |
| Tool Control | "Me" (self-controlled) | 1 | Direct control of productive assets indicates operational autonomy |
| Output Sale Control | "Me" (self-controlled) | 1 | Control over crop sales ensures ability to direct revenue toward loan repayment |
| Marital Status | Widowed or Divorced | 1 | Independent household management demonstrates self-reliance and financial autonomy |

## Survey Questions

- **Q9**: Gender (Dropdown: Male / Female)
- **Q9.1**: Are you the primary decision-maker in your household? (Yes / No; if No: Spouse / Relative / Other)
- **Q9.2**: Marital status (Single / Married / Widowed / Divorced / Separated)
- **Q24**: Who controls farm tools? (Me / Spouse / Other)
- **Q29**: Who controls the sale of farm output? (Me / Spouse / Other)

## Database Fields

- `gender` — Farmer's biological gender (Male/Female)
- `decision_making_role` — Role in household financial decisions (plot owner, manager, labor)
- `farm_tools` — Type of farm tools (Traditional / Modern) with control indicator
- `storageType` — Storage method for produced output with control indicator

## Related ML Features

- `gender` — Binary gender indicator used as a model feature
- `decision_making_role` — Categorical feature capturing the farmer's decision authority
- `asset_ownership` — Derived binary feature for productive asset control

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

In Ethiopian smallholder agriculture, women manage a significant share of farm operations but are systematically underrepresented in formal credit systems. Traditional credit scoring that uses gender as a binary variable often disadvantages women. The gender-intentional approach used here measures functional agency — the actual control a person has over productive resources and financial decisions — which is a much stronger predictor of loan repayment capacity than gender alone.

Research in agricultural microfinance consistently shows that women who control household financial decisions and productive assets have equal or better repayment rates compared to male borrowers. By scoring agency and control rather than just gender, the model identifies creditworthy women who would be rejected by conventional scoring systems. The additional point for widowed or divorced status recognizes that women managing households independently have demonstrated financial self-reliance.

This data point has High reliability (1.0) because the components are directly observable or verifiable during the survey — gender, marital status, and tool/output control can be confirmed by the field agent, reducing the risk of misreporting.
