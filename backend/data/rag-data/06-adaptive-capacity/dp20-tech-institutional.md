# DP20: Technology and Institutional Adaptive Capacity

## Summary

Technology and Institutional Adaptive Capacity measures the farmer's access to and adoption of modern agricultural technologies, digital tools, market linkages, and institutional support systems. This includes cooperative membership, aggregator connections, contract farming experience, agricultural extension services, and mobile/SMS farming tools. Farmers with stronger technology and institutional ties are more productive, better informed about markets, and better positioned to adapt to changing conditions — all of which support loan repayment.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: Average (0.66)
- **Framework Design Weight**: 10.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 10 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP20 is a weighted count of technology and institutional indicators:

```
DP20 = MIN(cap, SUM of:
  Cooperative_Membership (Q15) × 2 points
  Aggregator_Linkage (Q49) × 2 points
  Contract_Farming (Q42) × 1 point
  Extension_Access / Agricultural_Certificate (Q21/cert) × 1 point
  Farming_Apps_SMS (Q47) × 1 point
  Farm_Mechanization (Q28) × 1 point
  Weather_Advisory (Q50) × 1 point
)
```

## Technology and Institutional Components

| Component | Source | Weight | Rationale |
|-----------|--------|--------|-----------|
| Cooperative membership | Q15 | 2 | Market access, collective services, training programs |
| Buyer/aggregator linkage | Q49 | 2 | Guaranteed market outlet, price stability |
| Contract farming participation | Q42 | 1 | Structured agricultural agreement; proven discipline |
| Agricultural certificate/training | Certificate | 1 | Formal training improves practices and productivity |
| Mobile farming apps/SMS services | Q47 | 1 | Digital information access for market, weather, agronomy |
| Farm mechanization access | Q28 | 1 | Technology adoption for higher productivity |
| Weather advisory access | Q50 | 1 | Climate information for risk management |

## Survey Questions

- **Q15**: Are you a member of a farmer cooperative? (Yes/No)
- **Q24**: What type of farm tools do you use? (Traditional / Modern) — indicates mechanization level
- **Q26**: Farm size for target crop (contextual for mechanization assessment)
- **Q28**: Do you have access to farm mechanization services? (Yes/No)
- **Q42**: Do you have contract farming experience? (Yes/No)
- **Q47**: Do you use mobile apps or SMS services for farming information? (Yes/No)
- **Q49**: Are you linked to a buyer or aggregator? (Yes/No)
- **Q50**: Do you have access to weather advisory information? (Yes/No)

## Database Fields

- `Farm_Mechanization` — Access to mechanization services (boolean)
- `hasCooperativeAssociation` — Farmer cooperative membership (boolean)
- `agriculturalCertificate` — Agricultural training/certification (boolean)
- `certificateImage` — Uploaded certificate image for verification
- `weather_advisory_info` — Weather information source (SMS, extension, other)

## Related ML Features

- `farm_mechanization` — Categorical feature: manual, semi-mechanized, fully mechanized
- `asset_ownership` — Binary productive asset ownership (tools, equipment)
- `institutional_support_score` — Composite of 4 institutional flags

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Technology and institutional ties represent the farmer's connection to the modern agricultural economy. In Ethiopian agriculture, the gap between farmers with strong institutional connections and those operating in isolation is substantial — and growing.

**Cooperative membership** (weighted at 2 points) provides access to the formal agricultural value chain: bulk input purchasing at lower prices, collective marketing at higher prices, shared storage and transport infrastructure, and training programs. Cooperative members typically achieve 15-30% higher returns than non-members due to these collective benefits.

**Buyer/aggregator linkage** (2 points) addresses one of the most critical challenges for smallholder farmers: market access. Farmers linked to a reliable buyer have a guaranteed outlet for their harvest, reducing the risk of post-harvest losses due to inability to sell. This market certainty makes income projections more reliable and loan repayment more predictable.

**Mobile farming apps and SMS services** represent a transformative technology for smallholder farmers. Through SMS, farmers receive real-time market prices (enabling better selling decisions), weather forecasts (enabling better planting decisions), and agronomic advice (enabling better crop management). This information reduces uncertainty and improves decision-making across the entire farming cycle.

**Weather advisory access** is particularly important in a climate-vulnerable context. Farmers who receive weather information can adjust planting dates, variety selection, and input application to match expected conditions. This proactive climate adaptation directly reduces production risk.

**Farm mechanization** improves both productivity and timeliness. Mechanized land preparation allows farmers to plant at the optimal time (critical for rain-fed crops), and mechanized harvesting reduces post-harvest losses. Even rental access to tractors and threshers provides significant productivity benefits.

**Contract farming** demonstrates the farmer's ability to work within structured agricultural agreements — meeting quality standards, delivery schedules, and input repayment obligations. This discipline closely mirrors the commitment needed for loan repayment.

This data point has Average reliability (0.66) because most components involve verifiable institutional memberships or observable technology adoption, though some (like weather advisory access) depend on self-reporting.
