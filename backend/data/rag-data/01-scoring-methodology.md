# Data Point Framework and ML Scoring Methodology

## Overview

The Lersha Credit Scoring System uses a structured 20-data-point framework as the foundation for ML-based creditworthiness assessment. Each data point captures a specific dimension of a smallholder farmer's economic profile and was originally designed with two attributes:

- **Predictive Power**: How strongly the factor correlates with loan repayment behavior. Rated as Average (10) or High (15). This guided the original framework design but ML models learn their own feature importances from training data.
- **Reliability**: How accurately the data can be measured or verified. Rated as High (1.0), Average (0.5–0.66), or Low (0.25). This remains relevant as a data quality indicator.

The 20 data points feed into a feature engineering pipeline that produces 34 ML features. These features are used by an ensemble of machine learning models (XGBoost, Random Forest, CatBoost) to classify each farmer as **Eligible**, **Review**, or **Not Eligible**. SHAP (SHapley Additive exPlanations) values quantify how much each feature contributed to each individual prediction.

## Original Framework Design Weights

The table below shows the original design weights for each data point. These guided the initial framework construction but do not directly determine ML predictions — the models learn nonlinear relationships and feature interactions from historical training data, and actual feature importance varies per prediction (visible via SHAP values).

| DP# | Data Point | Category | Predictive Power | Reliability | Design Weight |
|-----|-----------|----------|-----------------|-------------|---------------|
| 1 | Gender Agency & Control | Demographics | Average (10) | High (1.0) | 10.0 |
| 2 | Age | Demographics | Average (10) | High (1.0) | 10.0 |
| 3 | Number of Dependents | Demographics | Average (10) | Average (0.5) | 5.0 |
| 4 | Family Homestead Type | Demographics | Average (10) | Average (0.5) | 5.0 |
| 5 | Cost of Target Crop Farming | Financial | High (15) | High (1.0) | 15.0 |
| 6 | Income from Target Crop Sales | Financial | High (15) | High (1.0) | 15.0 |
| 7 | Other Costs (Off-Farm/Household) | Financial | High (15) | Average (0.66) | 10.0 |
| 8 | Other Income (Off-Farm/Livestock) | Financial | High (15) | Average (0.66) | 10.0 |
| 9 | Loan Outstanding | Financial | Average (10) | Low (0.25) | 2.5 |
| 10 | Total Land Size Managed | Assets & Land | Average (10) | Average (0.5) | 5.0 |
| 11 | Land Title Availability | Assets & Land | Average (10) | High (1.0) | 10.0 |
| 12 | Land Size for Target Crop | Assets & Land | Average (10) | Average (0.5) | 5.0 |
| 13 | Farm Tool Ownership/Control | Assets & Land | Average (10) | Average (0.5) | 5.0 |
| 14 | Output Storage & Control | Assets & Land | Average (10) | Low (0.25) | 2.5 |
| 15 | Credit History | Credit & Climate | High (15) | Low (0.25) | 3.75 |
| 16 | Exposure to Extreme Events | Credit & Climate | High (15) | Low (0.25) | 3.75 |
| 17 | Sensitivity to Climate Variability | Credit & Climate | High (15) | Low (0.25) | 3.75 |
| 18 | Economic Adaptive Capability | Adaptive Capacity | High (15) | Average (0.66) | 10.0 |
| 19 | Social & Human Resource Adaptive | Adaptive Capacity | High (15) | Average (0.66) | 10.0 |
| 20 | Technology/Institutional Capacity | Adaptive Capacity | High (15) | Average (0.66) | 10.0 |

## How Predictions Are Made

### Step 1: Individual Data Point Calculation

Each data point receives a value based on its specific calculation formula. Most data points use one of these methods:

- **Tier-based calculation**: Direct mapping of values to brackets (e.g., Age: 47–59 years = 10 points)
- **Quartile-based calculation**: The farmer's value is compared against the population distribution using QUARTILE.INC, and the quartile rank determines the value
- **Binary/count-based calculation**: Sum of binary indicators (e.g., number of institutional memberships)
- **Lookup-based calculation**: Values derived from external reference tables (e.g., CropRef for cost/revenue)

### Step 2: Feature Engineering

The 34 ML features are derived from the 20 data point values and raw survey data. Key engineered features include:

- **net_income**: Total estimated income minus total estimated cost — primary creditworthiness indicator
- **yield_per_hectare**: Expected yield divided by farm size — productivity indicator
- **income_per_family_member**: Total income divided by family size — welfare indicator
- **input_intensity**: (Seeds + urea + DAP) divided by farm size — farming intensity proxy
- **institutional_support_score**: Sum of 4 binary flags (microfinance, cooperative, agri-cert, health-insurance)
- **agriculture_experience**: Log-transformed years of farming experience

### Step 3: ML Ensemble Prediction

Three machine learning models independently classify the farmer:

- **XGBoost**: Gradient-boosted decision trees optimized for tabular data
- **Random Forest**: Ensemble of decision trees with bagging for variance reduction
- **CatBoost**: Gradient boosting with native categorical feature handling

Each model outputs class probabilities via `predict_proba()` for the three eligibility classes (Eligible, Review, Not Eligible). The confidence score is the highest class probability.

### Step 4: SHAP Explainability

SHAP (SHapley Additive exPlanations) values are calculated for each prediction using TreeExplainer, quantifying how much each feature contributed to pushing the prediction toward or away from each class. The top 10 contributing features by absolute SHAP value are identified and used for explanation generation.

SHAP values reveal the actual feature importance for each individual farmer — this can differ significantly from the original framework design weights because the ML models capture nonlinear patterns and feature interactions learned from historical data.

## Design Weight Distribution by Category

| Category | Data Points | Total Design Weight | % of Total |
|----------|------------|-------------------|------------|
| Demographics | DP1–DP4 | 30.0 | ~19% |
| Financial Capacity | DP5–DP9 | 52.5 | ~34% |
| Assets & Land | DP10–DP14 | 27.5 | ~18% |
| Credit & Climate | DP15–DP17 | 11.25 | ~7% |
| Adaptive Capacity | DP18–DP20 | 30.0 | ~19% |
| **Total** | **DP1–DP20** | **~155** | **100%** |

The original framework design gave Financial Capacity (DP5–DP9) the highest weight at approximately 34%, reflecting that income, costs, and debt burden were expected to be the strongest predictors of repayment ability. ML model training on historical data validates and refines these relative importances — the models may discover that certain features are more or less predictive than originally assumed, and that feature interactions (e.g., income combined with family size) matter more than individual features alone.

## Gender-Intentional Design

The data point framework deliberately measures women's economic agency through multiple dimensions:

- **DP1** captures decision-making authority, tool control, and output sale control — not just binary gender
- **DP11** recognizes joint land titles with spouses
- **DP13** and **DP14** measure who controls farm tools and output sales (Me/Spouse/Other)
- **DP8** includes income from petty trade and food processing, which are predominantly women's economic activities

This design ensures that women who are de facto managers of farm operations receive credit assessments that reflect their actual capacity, even when formal land titles are held by male family members.

## Climate-Smart Integration

Climate dimensions are captured across DP16–DP20:

- **Exposure (DP16)**: How frequently the farmer experiences droughts, floods, and pest outbreaks
- **Sensitivity (DP17)**: How vulnerable the target crop is to rainfall variability
- **Adaptive Capacity (DP18–DP20)**: The farmer's ability to cope with and recover from climate shocks through economic resources, social networks, and technology access

This integration recognizes that in rain-fed agriculture, climate risk is often a stronger predictor of default than traditional financial metrics alone. The ML models learn how these climate dimensions interact with financial and asset features to influence creditworthiness.
