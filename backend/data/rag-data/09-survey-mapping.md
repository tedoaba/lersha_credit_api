# Survey Question to Data Point Mapping

## Overview

The Lersha Credit Scoring survey consists of 50 core questions plus 5 gender-intentional additions, organized into seven sections. Each question maps to one or more of the 20 Data Points (DP1–DP20) that form the credit scoring framework. This document provides the complete mapping from survey questions to data points, enabling understanding of how raw survey responses translate into credit scores.

## Section 1: General Farmer Demography (Q1–Q14)

| Q# | Question | Response Type | Data Point(s) | Purpose |
|----|----------|--------------|---------------|---------|
| Q1 | GPS geolocation | Auto-capture | — | Spatial verification; not scored |
| Q2 | Timestamp | Auto-capture | — | Data quality audit; not scored |
| Q3 | Field agent name | Text (mandatory) | — | Data quality traceability |
| Q4 | Woreda (district) | Text (mandatory) | — | Geographic classification |
| Q5 | Kebele (sub-district) | Text (mandatory) | — | Geographic classification |
| Q6 | Farmer first name | Text (mandatory) | — | Identification |
| Q7 | Farmer middle name | Text (mandatory) | — | Identification |
| Q8 | Farmer last name | Text (mandatory) | — | Identification |
| Q9 | Gender | Dropdown: Male/Female | **DP1** | Gender component of agency score |
| Q9.1 | Primary decision-maker? | Yes/No (+ Spouse/Relative/Other) | **DP1** | Decision-making authority (gender-intentional) |
| Q9.2 | Marital status | Single/Married/Widowed/Divorced/Separated | **DP1** | Independent management indicator (gender-intentional) |
| Q10 | Age | Numeric (18–99) | **DP2** | Age-based experience scoring |
| Q11 | Total family members | Numeric (0–20) | **DP3** | Household dependency burden |
| Q12 | Children under 12 | Numeric (0–10) | **DP3** | Non-productive dependent count |
| Q13 | Elderly members over 60 | Numeric (0–10) | **DP3** | Non-productive dependent count |
| Q14 | Homestead type | Traditional all-in-one / Traditional multiple rooms / Modern | **DP4** | Asset wealth proxy |

## Section 2: Socio-economic Engagement (Q15–Q21)

| Q# | Question | Response Type | Data Point(s) | Purpose |
|----|----------|--------------|---------------|---------|
| Q15 | Farmer cooperative membership | Yes/No | **DP19, DP20** | Social capital and institutional ties |
| Q16 | Farmer cluster membership | Yes/No | **DP19** | Peer learning network participation |
| Q17 | RUSACCO membership | Yes/No | **DP18** | Formal savings and credit access |
| Q18 | Community health insurance | Yes/No | **DP18** | Health shock protection |
| Q19 | Local EQUIB membership | Yes/No | **DP18** | Rotating savings group participation |
| Q20 | Local Edir membership | Yes/No | **DP19** | Community mutual aid society |
| Q21 | Leadership role in community org | Yes/No | **DP19** | Community trust and management ability |

## Section 3: Agronomic Activity (Q22–Q29)

| Q# | Question | Response Type | Data Point(s) | Purpose |
|----|----------|--------------|---------------|---------|
| Q22 | Total farmland managed (secure use rights) | Numeric (0–10 ha) | **DP10** | Total productive land base (gender-intentional: includes joint ownership) |
| Q23 | Land title certificate | Yes/No/Joint with spouse | **DP11** | Collateral and tenure security (gender-intentional: joint titles recognized) |
| Q24 | Farm tools type + control | Traditional/Modern + Me/Spouse/Other | **DP1, DP13** | Tool type and operational agency (gender-intentional) |
| Q25 | Target crop selection | Dropdown: Malt Barley/Wheat/Maize/Soybean/Tomato/Onion/Potatoes/Other | **DP5, DP6** | CropRef lookup key for cost/revenue |
| Q26 | Target crop farm size | Numeric (0–10 ha) | **DP5, DP6, DP12** | Area multiplier for cost/revenue calculations |
| Q27 | Target crop marketable yield % | Numeric (0–100%) | **DP6** | Revenue calculation component |
| Q28 | Other crops grown | Yes/No + up to 2 crop selections | **DP7, DP8** | Diversification assessment |
| Q28.1–Q28.5 | Other crop details | Crop type, area (ha), marketable yield % | **DP7, DP8** | Other crop cost/revenue calculations |
| Q29 | Output storage + control | Own storage/Aggregator/Commodity financing + Me/Spouse/Other | **DP1, DP14** | Post-harvest management and sales agency (gender-intentional) |

## Section 4: Livestock Income and Expense (Q30–Q35)

| Q# | Question | Response Type | Data Point(s) | Purpose |
|----|----------|--------------|---------------|---------|
| Q30 | Livestock ownership by type | Bulls/Cows/Mules/Horses/Goats/Sheep/Chickens | **DP8, DP18** | Livestock asset assessment |
| Q30.2–Q30.6 | Counts per livestock type | Numeric (0–20 each) | **DP8** | Livestock wealth quantification |
| Q32 | Monthly dairy/milk income | Numeric (0–10,000 ETB) | **DP8** | Regular livestock cash flow |
| Q33 | Off-farm income activities | Yes/No | **DP8, DP18** | Income diversification indicator |
| Q33.1 | Monthly salary income | Numeric (ETB) | **DP8** | Formal employment income |
| Q33.2 | Monthly remittance income | Numeric (ETB) | **DP8** | Family transfer income |
| Q33.3 | Monthly rental income | Numeric (ETB) | **DP8** | Property/equipment rental income |
| Q34 | Regular monthly expenses | Yes/No | **DP7** | Household expense indicator |
| Q34.1 | Monthly student support amount | Numeric (ETB) | **DP7** | Education expense burden |
| Q34.2 | Monthly land rent amount | Numeric (ETB) | **DP7** | Land rental cost |
| Q34.3 | Monthly vet service expenses | Numeric (ETB) | **DP7** | Livestock maintenance cost |
| Q35 | Trading/food processing income | Yes/No + amount (ETB) | **DP8** | Women's economic activities (gender-intentional) |

## Section 5: Credit History Proxy (Q36–Q42)

| Q# | Question | Response Type | Data Point(s) | Purpose |
|----|----------|--------------|---------------|---------|
| Q36 | Informal credit use (family/friends/local) | Yes/No | **DP15** | Informal borrowing patterns |
| Q37 | Formal credit access (last 12 months) | Yes/No | **DP15** | Formal financial engagement |
| Q38 | Loan repayment difficulty | Yes/No | **DP15** | Repayment track record |
| Q39 | Currently has outstanding loan | Yes/No | **DP9, DP15** | Current debt indicator |
| Q40 | Outstanding loan amount | Numeric (0–100,000 ETB) | **DP9** | Debt burden quantification |
| Q41 | Savings account (bank/MFI/RUSACCO) | Yes/No | **DP15, DP18** | Financial discipline and formal savings |
| Q42 | Contract farming experience | Yes/No | **DP15, DP20** | Structured agriculture participation |

## Section 6: Climate Risk Profile (Q43–Q45)

| Q# | Question | Response Type | Data Point(s) | Purpose |
|----|----------|--------------|---------------|---------|
| Q43.3 | Pest/disease frequency change | Never/Rarely/Sometimes/Frequently/Very Frequently | **DP16** | Biotic climate risk exposure |
| Q43.4 | Rainfall variability impact on yield | Not at all/Slightly/Moderately/Very/Extremely | **DP16, DP17** | Climate impact assessment |
| Q43.5 | Crop sensitivity to rainfall timing | Not sensitive to Extremely sensitive | **DP17** | Crop-specific climate vulnerability |
| Q43.6 | Climate impact on farming income | Not at all to Extremely | **DP17** | Economic impact of climate variability |
| Q44 | Crop insurance for climate risk | Yes/No | **DP18** | Financial climate protection |
| Q45 | Irrigation/water reserve access | Yes/No | **DP18** | Physical climate adaptation |

## Section 7: Adaptive Capacity (Q46–Q50)

| Q# | Question | Response Type | Data Point(s) | Purpose |
|----|----------|--------------|---------------|---------|
| Q46 | Shares farming knowledge with others | Yes/No | **DP19** | Knowledge network participation |
| Q47 | Uses mobile farming apps/SMS services | Yes/No | **DP20** | Digital technology adoption |
| Q28 (repeated) | Farm mechanization access | Yes/No | **DP20** | Technology adoption level |
| Q49 | Linked to buyer/aggregator | Yes/No | **DP20** | Market access and institutional ties |
| Q50 | Weather advisory access | Yes/No | **DP20** | Climate information access |

## Data Point Source Summary

This table shows which survey questions feed into each data point:

| Data Point | Source Questions | External Data |
|-----------|----------------|---------------|
| DP1: Gender Agency | Q9, Q9.1, Q9.2, Q24, Q29 | — |
| DP2: Age | Q10 | — |
| DP3: Dependents | Q11, Q12, Q13 | — |
| DP4: Homestead | Q14 | — |
| DP5: Cost Target Crop | Q25, Q26 | CropRef (Total Cost/Ha) |
| DP6: Income Target Crop | Q25, Q26, Q27 | CropRef (Revenue/Ha) |
| DP7: Other Costs | Q28.2, Q28.4, Q34.1, Q34.2, Q34.3 | CropRef (for other crop costs) |
| DP8: Other Income | Q28.3, Q28.5, Q30–Q30.6, Q32, Q33.1–Q33.3, Q35 | CropRef (for other crop revenue) |
| DP9: Loan Outstanding | Q39, Q40 | — |
| DP10: Land Managed | Q22 | — |
| DP11: Land Title | Q23 | — |
| DP12: Land Target Crop | Q26 | — |
| DP13: Farm Tools | Q24, Q28 (mechanization) | — |
| DP14: Output Storage | Q29 | — |
| DP15: Credit History | Q36, Q37, Q38, Q39, Q40, Q41, Q42 | — |
| DP16: Climate Exposure | Q43.3, Q43.4 | — |
| DP17: Climate Sensitivity | Q43.4, Q43.5, Q43.6 | — |
| DP18: Economic Adaptive | Q17, Q18, Q19, Q33, Q41, Q44, Q45, Q49 | — |
| DP19: Social Adaptive | Q15, Q16, Q20, Q21, Q46 | — |
| DP20: Tech/Institutional | Q15, Q24, Q26, Q28, Q42, Q47, Q49, Q50 | — |

## Gender-Intentional Questions

Five questions were specifically added or modified to capture women's economic agency:

1. **Q9.1** (Primary decision-maker) — Captures actual decision authority, not assumed from gender
2. **Q9.2** (Marital status) — Identifies widowed/divorced women managing households independently
3. **Q23** ("Joint with spouse" option) — Recognizes women's land rights through joint titling
4. **Q24/Q29** ("Me/Spouse/Other" control) — Measures who controls productive assets and output sales
5. **Q35** (Trading/food processing income) — Captures informal income streams predominantly managed by women
