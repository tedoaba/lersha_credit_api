# Lersha Credit Scoring System — Overview

## What Is Lersha Credit Scoring?

The Lersha Credit Scoring System is an agricultural credit assessment platform designed specifically for Ethiopian smallholder farmers. It evaluates a farmer's creditworthiness using a structured framework of 20 Data Points (DP1–DP20) that capture financial capacity, asset ownership, climate risk exposure, and adaptive capabilities. The system classifies each farmer into one of three eligibility categories: **Eligible**, **Review**, or **Not Eligible**.

The model is **gender-intentional** — it measures women's actual economic agency and decision-making control rather than using gender as a simple binary variable. It is also **climate-smart** — it integrates climate risk exposure, crop sensitivity, and adaptive capacity into the scoring framework, recognizing that climate variability is a primary driver of loan default risk in rain-fed agriculture.

## Target Population

Ethiopian smallholder farmers, typically cultivating 0.5–10 hectares of land across diverse agro-ecological zones. These farmers grow staple crops (wheat, maize, teff, malt barley), cash crops (coffee, sesame), horticultural crops (tomato, onion, potatoes), and legumes (soybean, haricot bean, chickpea). Many also derive income from livestock, off-farm activities, and informal savings groups.

## How the Scoring Works

1. **Data Collection**: A field agent conducts a structured survey of 50 questions covering demographics, farm operations, income/expenses, credit history, and climate exposure.
2. **External Data Integration**: Crop-specific cost and revenue data is sourced from standardized reference tables (CropRef), removing subjectivity from financial projections.
3. **20 Data Points Calculated**: Survey responses and external data are combined to compute 20 scored dimensions (DP1–DP20), each with defined predictive power and reliability weights.
4. **ML Ensemble Prediction**: An ensemble of machine learning models (XGBoost, Random Forest, CatBoost) trained on the 20 DP scores plus 34 derived features produces a credit eligibility prediction.
5. **SHAP Explainability**: SHAP values identify which features most influenced the prediction for each individual farmer.
6. **RAG Explanation**: A retrieval-augmented generation pipeline provides natural-language explanations of the credit decision, grounded in the knowledge base.

## Three Decision Outcomes

| Outcome | Meaning |
|---------|---------|
| **Eligible** | Farmer meets creditworthiness criteria; recommended for loan approval |
| **Review** | Borderline case requiring manual assessment by a loan officer |
| **Not Eligible** | Farmer does not currently meet credit requirements; may reapply after addressing identified gaps |

## Key Design Principles

- **Objective Data Priority**: Cost and revenue projections use externally validated crop reference data, not farmer self-reports alone.
- **Data Quality Awareness**: Data points are rated for reliability (how accurately they can be measured). Lower-reliability inputs (e.g., self-reported credit history) were given reduced design weights in the original framework. The ML models also learn to appropriately weight features based on their predictive value in training data.
- **Gender-Intentional**: Captures functional agency (decision-making, tool control, output sale control) rather than binary gender.
- **Climate-Smart**: Integrates exposure, sensitivity, and adaptive capacity dimensions aligned with climate vulnerability frameworks.
- **Diversification Rewarded**: Multiple income streams (off-farm, livestock, trading) and social capital (cooperative membership, savings groups) increase scores.

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **ETB** | Ethiopian Birr — the national currency of Ethiopia |
| **Quintal** | A unit of weight equal to 100 kilograms, commonly used for crop yields in Ethiopia |
| **Hectare (ha)** | A unit of land area equal to 10,000 square meters (approximately 2.47 acres) |
| **Woreda** | An administrative district in Ethiopia, equivalent to a county |
| **Kebele** | The smallest administrative unit in Ethiopia, equivalent to a neighborhood or village |
| **EQUIB (Equb)** | A traditional Ethiopian rotating savings and credit association where members contribute fixed amounts and take turns receiving the pooled sum |
| **RUSACCO** | Rural Savings and Credit Cooperative Organization — a formal community-based financial institution providing savings and small loans |
| **Edir (Idir)** | A traditional Ethiopian community mutual aid association, originally for funeral expenses, now often providing broader social safety net support |
| **MFI** | Microfinance Institution — a financial organization providing small loans and savings services to unbanked populations |
| **DAP** | Diammonium Phosphate — a widely used phosphorus-based fertilizer |
| **NPS** | Nitrogen, Phosphorus, Sulfur — a blended fertilizer common in Ethiopian agriculture |
| **Urea** | A nitrogen-based fertilizer (46% nitrogen) used to promote vegetative growth |
| **SHAP** | SHapley Additive exPlanations — a method for explaining individual ML predictions by quantifying each feature's contribution |
| **Data Point (DP)** | One of 20 scored dimensions in the credit scoring framework, each capturing a specific aspect of creditworthiness |
| **Predictive Power** | A framework design rating indicating how strongly a data point correlates with loan repayment behavior (Average = 10, High = 15). Note: ML models learn their own feature importances from training data; actual importance per prediction is quantified by SHAP values |
| **Reliability** | How accurately the data point can be measured or verified (High = 1.0, Average = 0.5–0.66, Low = 0.25). Used as a data quality indicator |
| **Framework Design Weight** | The original design weight for a data point (Predictive Power x Reliability). Guided the initial framework design but does not directly determine ML model predictions |
| **CropRef** | The external crop reference data table containing standardized costs, yields, and prices per hectare for each supported crop |
| **Class Probability** | The ML model's predicted probability for each eligibility class (Eligible, Review, Not Eligible), summing to 1.0. The class with the highest probability becomes the prediction |
| **Confidence Score** | The highest class probability value, indicating how confident the model is in its prediction |

## Data Point Categories

The 20 Data Points are organized into five domains:

1. **Demographics (DP1–DP4)**: Gender agency, age, dependents, homestead type
2. **Financial Capacity (DP5–DP9)**: Target crop cost/revenue, other costs/income, outstanding loans
3. **Assets and Land (DP10–DP14)**: Land managed, land title, crop allocation, farm tools, output storage
4. **Credit History and Climate Risk (DP15–DP17)**: Credit behavior, climate exposure, crop sensitivity
5. **Adaptive Capacity (DP18–DP20)**: Economic, social/human resource, and technology/institutional capabilities
