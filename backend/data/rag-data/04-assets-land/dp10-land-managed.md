# DP10: Total Land Size Managed

## Summary

Total Land Size Managed measures the total farmland under the farmer's operational control, including owned land, family land, and rented land. Larger land holdings indicate greater production potential and asset wealth. This data point captures the farmer's productive base — the fundamental resource from which agricultural income is generated. The scoring recognizes both formal ownership and operational use rights, including joint management arrangements common in Ethiopian households.

## Data Point Characteristics

- **Predictive Power**: Average (10)
- **Reliability**: Average (0.5)
- **Framework Design Weight**: 5.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 5 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

Total land managed is reported directly from Q22 and scored using quartile-based ranking against the population distribution:

```
Land_Managed = own_farmland_size + family_farmland_size + rented_farm_land
Score = QUARTILE-based scoring of Land_Managed
```

Farmers with more land under management receive higher scores.

## Scoring Tiers

| Quartile | Land Size | Score Direction | Rationale |
|----------|----------|-----------------|-----------|
| 1st Quartile | Smallest holdings | Lowest score | Limited production potential |
| 2nd Quartile | Below average | Below-average score | Constrained but viable |
| 3rd Quartile | Above average | Above-average score | Strong production base |
| 4th Quartile | Largest holdings | Highest score (5) | Maximum production potential and asset wealth |

## Survey Questions

- **Q22**: What is the total farmland you manage or have secure use rights to? (Numeric, range 0–10 hectares)
  - Note: This question is gender-intentional — it asks about "manage or have secure use rights" rather than strictly "own," recognizing that women often manage land they do not formally own.

## Database Fields

- `total_farmLand_size` — Total farm size in hectares for the season
- `own_farmland_size` — Hectares of personally owned land
- `family_farmland_size` — Hectares of family-owned land the farmer manages
- `rented_farm_land` — Hectares of rented land

## Related ML Features

- `farmsizehectares` — Total operated farm area in hectares (primary feature)
- `own_farmland_size` — Size of owned farmland, indicating asset ownership
- `family_farmland_size` — Total family-controlled farmland
- `rented_farm_land` — Size of rented land, indicating operational scale beyond owned assets

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Land is the most fundamental productive asset in smallholder agriculture. The total area under a farmer's management directly determines their maximum production potential — a farmer managing 3 hectares can produce roughly three times the output of a farmer managing 1 hectare, all else being equal. This production potential translates to income-generating capacity and, ultimately, loan repayment ability.

The distinction between owned, family, and rented land is important. Owned land represents a stable, long-term asset that provides collateral value. Family land indicates access to resources through kinship networks but may be less secure. Rented land expands the farmer's production capacity but introduces ongoing rental costs (captured in DP7) and tenure uncertainty.

In the Ethiopian context, the gender-intentional framing of Q22 is significant. Many women farmers manage family land that is formally titled to male relatives. By asking about "manage or have secure use rights" rather than strict ownership, the survey captures the actual productive base available to the farmer, regardless of formal title status. This approach prevents systematically penalizing women who are de facto farm managers but lack formal land documentation.

This data point has Average reliability (0.5) because while land size is a relatively stable and verifiable characteristic, self-reported area estimates can be imprecise — farmers often report land in local units that must be converted to hectares, and boundary estimates may vary. GPS-verified parcel data, when available, improves accuracy.
