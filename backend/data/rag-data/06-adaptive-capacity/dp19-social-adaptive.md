# DP19: Social and Human Resource Adaptive Capabilities

## Summary

Social and Human Resource Adaptive Capabilities measures the farmer's social capital, community networks, and human resource capacity for adapting to challenges. This includes leadership roles, knowledge-sharing participation, Edir membership, farmer cluster membership, and cooperative engagement. Strong social networks provide information flow, mutual support during crises, access to collective resources, and behavioral accountability — all of which contribute to loan repayment reliability.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: Average (0.66)
- **Framework Design Weight**: 10.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 10 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP19 is a weighted count of social and human resource indicators, capped at a maximum:

```
DP19 = MIN(cap, SUM of:
  Leadership_Role (Q21) × 2 points
  Knowledge_Sharing (Q46) × 1 point
  Edir_Membership (Q20) × 1 point
  Cluster_Membership (Q16) × 1 point
  Cooperative_Membership (Q15) × 2 points
)
```

## Social Capital Components

| Component | Source | Weight | Rationale |
|-----------|--------|--------|-----------|
| Leadership role in community organization | Q21 | 2 | Demonstrates financial maturity, community trust, and management capability |
| Knowledge sharing on farming practices | Q46 | 1 | Indicates openness to learning, information flow, and peer networks |
| Edir membership | Q20 | 1 | Social safety net; funeral/emergency mutual aid |
| Farmer cluster membership | Q16 | 1 | Peer learning group; access to extension services |
| Cooperative membership | Q15 | 2 | Formal organization; market access, collective bargaining, input supply |

## Survey Questions

- **Q15**: Are you a member of a farmer cooperative? (Yes/No)
- **Q16**: Are you a member of a farmer cluster (peer learning group)? (Yes/No)
- **Q20**: Are you a member of a local Edir (mutual aid society)? (Yes/No)
- **Q21**: Do you hold a leadership role in any community organization? (Yes/No)
- **Q46**: Do you share farming knowledge with other farmers? (Yes/No)

## Database Fields

- `hasCooperativeAssociation` — Farmer cooperative membership (boolean)
- `hasLocalEdir` — Edir (mutual aid society) membership (boolean)
- `holdsLeadershipRole` — Leadership position in a community organization (boolean)

## Related ML Features

- `holdsleadershiprole` — Binary feature: holds a leadership role in a community organisation (1=yes, 0=no)
- `haslocaledir` — Binary feature: local cooperative membership (1=yes, 0=no)
- `institutional_support_score` — Composite score including cooperative and other institutional memberships

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Social capital is a powerful but often overlooked predictor of credit behavior in smallholder agriculture. Farmers embedded in strong social networks benefit from multiple creditworthiness-enhancing mechanisms:

**Information access**: Farmers in cooperatives and clusters receive timely information about input availability, market prices, weather forecasts, and pest management. This information advantage translates directly to better farm management decisions, higher yields, and more predictable income — all of which support loan repayment.

**Mutual support**: Edir membership provides a community safety net for emergencies. When a member faces a crisis (illness, death, crop failure), the Edir provides financial and labor support. This safety net prevents individual shocks from escalating into loan defaults. A farmer who can rely on community support during a difficult period is more likely to maintain loan repayment than an isolated farmer facing the same challenge alone.

**Accountability and reputation**: Farmers who hold leadership roles and participate actively in community organizations have reputational capital to protect. Defaulting on a loan damages their standing in the community, creating a social incentive for responsible financial behavior. Leadership roles also indicate management skills, organizational ability, and community trust — traits that predict responsible loan management.

**Cooperative benefits**: Farmer cooperatives provide access to bulk input purchasing (lower costs), collective marketing (better prices), shared equipment, and training programs. These tangible benefits improve farm profitability and, consequently, loan repayment capacity. Cooperative members also have access to cooperative credit programs that build financial literacy.

**Knowledge sharing** indicates that the farmer is both willing to learn from others and able to teach — a sign of experience, openness to innovation, and integration into the farming community's information networks.

This data point has Average reliability (0.66) because memberships in cooperatives, Edir, and clusters are verifiable through organizational records, while knowledge sharing and leadership are somewhat more subjective self-assessments.
