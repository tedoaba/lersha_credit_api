# DP18: Economic Adaptive Capability

## Summary

Economic Adaptive Capability measures the farmer's ability to cope with and recover from economic and climate shocks through financial mechanisms and resources. This includes crop insurance, irrigation or water reserve access, off-farm income sources, health insurance, savings participation (EQUIB, RUSACCO), and linkages to buyers or aggregators. A higher count of these adaptive mechanisms indicates greater economic resilience — the farmer can absorb shocks without defaulting on loan obligations.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: Average (0.66)
- **Framework Design Weight**: 10.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 10 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP18 is a weighted count of economic adaptive mechanisms, capped at 10 points:

```
DP18 = MIN(10, SUM of:
  Crop_Insurance (Q44) × 3 points
  Water_Reserve_Access (Q45) × 2 points
  Off_Farm_Income (Q33) × 1 point
  Health_Insurance (Q18) × 1 point
  EQUIB_Membership (Q19) × 1 point
  RUSACCO_Membership (Q17) × 1 point
  Buyer_Linkage (Q49) × 1 point
)
```

Different mechanisms receive different weights reflecting their relative importance for economic resilience.

## Adaptive Mechanisms

| Mechanism | Source | Weight | Rationale |
|-----------|--------|--------|-----------|
| Crop insurance | Q44 | 3 | Direct financial protection against crop failure; highest impact |
| Water reserve/irrigation access | Q45 | 2 | Reduces dependence on rainfall; high impact on yield stability |
| Off-farm income | Q33 | 1 | Alternative cash flow during crop failure |
| Community health insurance | Q18 | 1 | Prevents health shocks from depleting farm income |
| EQUIB membership | Q19 | 1 | Access to rotating savings pool for emergency liquidity |
| RUSACCO membership | Q17 | 1 | Formal savings and small loan access |
| Buyer/aggregator linkage | Q49 | 1 | Guaranteed market access reduces price risk |

## Survey Questions

- **Q17**: Are you a member of a RUSACCO (Rural Savings and Credit Cooperative)? (Yes/No)
- **Q18**: Do you have community health insurance? (Yes/No)
- **Q19**: Are you a member of a local EQUIB (rotating savings group)? (Yes/No)
- **Q33**: Do you have off-farm income activities? (Yes/No)
- **Q44**: Do you have crop insurance for climate risk? (Yes/No)
- **Q45**: Do you have access to irrigation or water reserves? (Yes/No)
- **Q49**: Are you linked to a buyer or aggregator? (Yes/No)

## Database Fields

- `access_water` — Water/reserve water access (boolean)
- `hasRuSACCO` — RUSACCO membership (boolean)
- `hasCommunityHealthInsurance` — Health insurance access (boolean)
- `HasParticipateInEQUIB` — EQUIB participation (boolean)
- `EQUIBContributionPerMonth` — Monthly EQUIB contribution (ETB)
- `HasReceivedEQUIB` — Whether EQUIB payout has been received (boolean)
- `HasOffFarmIncome` — Off-farm income indicator (boolean)

## Related ML Features

- `water_reserve_access` — Binary feature for irrigation/water access (1=yes, 0=no)
- `hasrusacco` — RUSACCO membership indicator
- `institutional_support_score` — Composite score of 4 binary institutional flags: microfinance membership, cooperative membership, agricultural certification, health insurance

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Economic adaptive capability is what separates farmers who can survive a bad season from those who default. In Ethiopian agriculture, where climate shocks affect millions of farmers simultaneously, the ability to cope economically determines whether a temporary production shortfall becomes a permanent credit default.

**Crop insurance** receives the highest weight (3 points) because it provides direct financial compensation for crop losses. A farmer with crop insurance can repay a portion of their loan even in a total crop failure year, dramatically reducing default risk from the lender's perspective.

**Water reserve access** (2 points) is the next most impactful mechanism because irrigation or supplemental water can mean the difference between a full harvest and a complete crop failure during drought. Farmers with water reserves can sustain crop production through dry spells that devastate rain-dependent neighbors.

**EQUIB participation** is particularly significant in the Ethiopian context. EQUIB (rotating savings groups) function as informal insurance — when a member faces a financial shock, the group can adjust the payout schedule to provide emergency liquidity. Regular EQUIB contributions (captured in the database) also demonstrate financial discipline, as members must maintain consistent payments to remain in good standing.

**Health insurance** may seem unrelated to agricultural credit, but health shocks are one of the leading causes of household financial distress in rural Ethiopia. A farmer without health insurance who faces a medical emergency may divert all available cash (including loan repayment funds) to healthcare costs. Community health insurance prevents this cascade from farm income to medical bills.

This data point has Average reliability (0.66) because most components are verifiable memberships (RUSACCO, EQUIB, insurance) that can be confirmed through institutional records, though some (like water access) depend on self-reporting.
