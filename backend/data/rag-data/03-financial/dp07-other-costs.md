# DP7: Other Costs (Off-Farm and Household Expenses)

## Summary

Other Costs captures the farmer's non-target-crop financial obligations, including household expenses, other crop production costs, and regular monthly outlays. This data point measures the financial drain on household resources beyond the primary farming activity. Lower other costs indicate more disposable income available for loan repayment. Unlike DP5 (target crop cost), these costs are self-reported, resulting in a lower reliability rating.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: Average (0.66)
- **Framework Design Weight**: 10.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 10 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP7 aggregates multiple expense categories from the survey:

```
Other_Costs = Student_Support_Monthly + Land_Rent_Monthly + Vet_Expenses_Monthly + Other_Crop_1_Cost + Other_Crop_2_Cost
```

Where other crop costs are calculated similarly to DP5:
```
Other_Crop_Cost = XLOOKUP(Other_Crop, CropRef!Crop_Column, CropRef!Total_Cost_Per_Ha) × Other_Crop_Area_Ha
```

The total is then quartile-scored against the population distribution — farmers with lower other costs receive higher scores.

## Cost Components

| Component | Source | Description |
|-----------|--------|-------------|
| Student support/remittance | Q34.1 | Monthly payments for children's education (ETB) |
| Land rent | Q34.2 | Monthly land rental payments (ETB) |
| Veterinary expenses | Q34.3 | Monthly livestock healthcare costs (ETB) |
| Other crop 1 costs | Q28.2 | Production cost for first diversification crop |
| Other crop 2 costs | Q28.4 | Production cost for second diversification crop |

## Scoring Tiers

| Quartile | Cost Level | Score Direction | Rationale |
|----------|-----------|-----------------|-----------|
| 1st Quartile | Lowest other costs | Highest score (10) | Minimal non-farm financial obligations |
| 2nd Quartile | Below average | Above-average score | Manageable expense base |
| 3rd Quartile | Above average | Below-average score | Significant competing financial demands |
| 4th Quartile | Highest other costs | Lowest score | Heavy expense burden reduces repayment capacity |

## Survey Questions

- **Q28**: Do you grow other crops besides your target crop? (Yes/No)
- **Q28.1–Q28.5**: Other crop selection, area, and marketable yield percentage (up to 2 crops)
- **Q34**: Do you have regular monthly expenses? (Yes/No)
- **Q34.1**: Monthly student support/remittance amount (ETB)
- **Q34.2**: Monthly land rent amount (ETB)
- **Q34.3**: Monthly veterinary service expenses (ETB)

## Database Fields

- `monthlyExpenses` — Whether the farmer has regular monthly expenses (boolean)
- `studentRemittanceAmount` — Monthly education support payments (ETB)
- `landRentAmount` — Monthly land rental cost (ETB)
- `rented_farm_land` — Size of rented farmland in hectares

## Related ML Features

- `total_estimated_cost` — Sum of production expenses and estimated total costs. This aggregate feature incorporates DP7 components alongside target crop costs.
- `net_income` — Total income minus total cost. DP7 costs reduce net income, directly impacting this primary creditworthiness indicator.
- `rented_farm_land` — Size of rented land, which correlates with land rent expenses.

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

A farmer's total financial picture extends well beyond their primary crop. Regular monthly obligations like school fees, land rent, and livestock care represent fixed costs that must be paid regardless of harvest outcomes. These obligations compete directly with loan repayment for the farmer's limited cash flow.

In Ethiopian smallholder contexts, student remittances are particularly significant — families often prioritize children's education over all other financial obligations, including loan repayment. A farmer sending multiple children to secondary school or university may face monthly education costs that rival their total farm income during off-seasons.

Land rent is another critical cost component. Farmers who rent a significant portion of their cultivated land face ongoing rental obligations that reduce their net income. Unlike owned land (which is a one-time investment), rented land represents a recurring cash outflow that must be sustained regardless of crop performance.

This data point has Average reliability (0.66) because household expenses are self-reported and farmers may understate their obligations to appear more creditworthy. However, the inclusion of CropRef-based calculations for other crop costs adds an objective component that partially offsets the self-reporting bias.
