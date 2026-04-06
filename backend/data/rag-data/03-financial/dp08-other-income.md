# DP8: Other Income (Off-Farm and Livestock)

## Summary

Other Income captures the farmer's revenue streams beyond their primary target crop, including livestock income (dairy, animal sales), off-farm employment (salary, remittances, rental income), petty trade, food processing, and revenue from secondary crops. Diversified income sources significantly reduce credit risk because the farmer is not entirely dependent on a single crop's success for loan repayment. Higher diversified income earns a higher score.

## Data Point Characteristics

- **Predictive Power**: High (15)
- **Reliability**: Average (0.66)
- **Framework Design Weight**: 10.0
- *Note: The ML models (XGBoost, Random Forest, CatBoost) learn their own feature importances from historical data. This weight reflects the original framework design intent, not the model's learned importance for any individual prediction. Actual feature impact varies per farmer and is quantified by SHAP values.*
- **Maximum Raw Score**: 10 points (after weighting)

## How It Is Calculated

This data point is calculated from survey data as follows. The resulting value feeds into the ML feature engineering pipeline, where it contributes to one or more of the 34 features used by the ensemble models.

DP8 aggregates multiple income streams from the survey:

```
Other_Income = Dairy_Income_Monthly + Salary_Income_Monthly + Remittance_Income_Monthly + Rental_Income_Monthly + Trading_Income_Monthly + Other_Crop_1_Revenue + Other_Crop_2_Revenue
```

Where other crop revenues are calculated using CropRef data:
```
Other_Crop_Revenue = XLOOKUP(Other_Crop, CropRef!Crop_Column, CropRef!Revenue_Per_Ha) × Other_Crop_Area_Ha
```

The total is quartile-scored — farmers with higher diversified income receive higher scores.

## Income Components

| Component | Source | Description |
|-----------|--------|-------------|
| Dairy/milk income | Q32 | Monthly income from milk and dairy products (0–10,000 ETB) |
| Salary income | Q33.1 | Monthly formal employment income (ETB) |
| Remittance income | Q33.2 | Monthly transfers from family abroad or in cities (ETB) |
| Rental income | Q33.3 | Monthly income from property or equipment rental (ETB) |
| Trading/processing income | Q35 | Small-scale trading and food processing revenue (ETB) — gender-intentional |
| Other crop 1 revenue | Q28.2–Q28.3 | Revenue from first diversification crop |
| Other crop 2 revenue | Q28.4–Q28.5 | Revenue from second diversification crop |
| Livestock sales | Q30–Q30.6 | Planned animal sale income |

## Scoring Tiers

| Quartile | Income Level | Score Direction | Rationale |
|----------|-------------|-----------------|-----------|
| 1st Quartile | Lowest other income | Lowest score | Highly dependent on single crop; concentrated risk |
| 2nd Quartile | Below average | Below-average score | Limited diversification |
| 3rd Quartile | Above average | Above-average score | Good income diversification |
| 4th Quartile | Highest other income | Highest score (10) | Strong income diversification; resilient cash flow |

## Survey Questions

- **Q28**: Do you grow other crops? (Yes/No), with up to 2 crop selections and areas
- **Q30**: Livestock ownership by type (Bulls/Cows/Mules/Horses/Goats/Sheep/Chickens)
- **Q30.2–Q30.6**: Counts per livestock type (range 0–20 each)
- **Q32**: Monthly milk/dairy income (Numeric, 0–10,000 ETB)
- **Q33**: Do you have off-farm income activities? (Yes/No)
- **Q33.1**: Monthly salary income (ETB)
- **Q33.2**: Monthly remittance income (ETB)
- **Q33.3**: Monthly rental income (ETB)
- **Q35**: Do you earn income from small-scale trading or food processing? (Yes/No + amount in ETB)

## Database Fields

- `milkIncomePerMonth` — Monthly dairy income in ETB
- `HasOffFarmIncome` — Whether farmer has off-farm income (boolean)
- `typeOffFarmIncome` — Type: salary, remittance, rental
- `salaryIncomePerMonth`, `remittanceIncomePerMonth`, `rentalIncomePerMonth` — Monthly amounts
- `totalLivestock` — Total animal count
- `livestock_types` — Types of animals owned
- `plannedSellPricePerAnimal` — Expected sale price per livestock

## Related ML Features

- `total_estimated_income` — Sum of primary farm income and all other income activities
- `income_per_family_member` — Total income divided by family size (welfare indicator)
- `net_income` — Total income minus total cost (primary creditworthiness indicator)

## Role in ML Pipeline

The data point values and related survey responses feed into the feature engineering pipeline, which derives the ML features listed above. The ML ensemble models (XGBoost, Random Forest, CatBoost) learn nonlinear relationships and interactions between these features and all other features in the model. Unlike fixed scoring weights, the actual importance of this data point varies per prediction and is quantified by SHAP (SHapley Additive exPlanations) values, which show exactly how much each feature pushed the prediction toward Eligible, Review, or Not Eligible for each individual farmer.

## Why This Matters for Credit Scoring

Income diversification is one of the strongest predictors of loan repayment in rain-fed agriculture. A farmer who relies solely on their target crop faces catastrophic risk — a single drought, pest outbreak, or price collapse can eliminate their entire income and make loan repayment impossible. In contrast, a farmer with dairy income, off-farm salary, remittances, and secondary crops can sustain loan payments even when their primary crop underperforms.

The gender-intentional design of DP8 is particularly important. Question Q35 explicitly captures income from small-scale trading and food processing, which are predominantly women's economic activities in Ethiopian rural communities. Traditional credit scoring systems often overlook these informal income streams, systematically understating women's true income-generating capacity. By including trading and processing income, the model recognizes the full economic contribution of women farmers.

Livestock income (dairy, animal sales) serves as both a regular income stream and an emergency liquidity source. Farmers who own cows generating monthly milk income have a steady cash flow that can service loan repayments during the crop growing season before harvest income arrives. This "bridge income" is critical for maintaining payment schedules aligned with agricultural production cycles.

This data point has Average reliability (0.66) because income amounts are self-reported and may be subject to estimation error, particularly for informal activities like trading. However, livestock ownership and off-farm employment can be partially verified by field agents, and CropRef-based revenue calculations for secondary crops add objectivity.
