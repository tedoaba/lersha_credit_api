# ML Model Feature Definitions

## Overview

The Lersha Credit Scoring ML models (XGBoost, Random Forest, CatBoost) use 34 features derived from the 20 Data Points and survey data. Each feature represents a specific dimension of the farmer's profile that the models use to predict creditworthiness. This document provides expanded definitions for each feature, grouped by domain.

## Feature Categories

The 34 features are organized into seven domains:
1. **Demographic Features** (4 features)
2. **Asset and Access Features** (7 features)
3. **Institutional Features** (3 features)
4. **Farm Operation Features** (5 features)
5. **Input Features** (4 features)
6. **Yield Features** (3 features)
7. **Income and Financial Features** (4 features)
8. **Target Variable** (1 feature)

---

## 1. Demographic Features

### gender
**Type**: Categorical (Male / Female)
**Description**: The farmer's biological gender. In the credit scoring context, gender is not used as a simple binary predictor — instead, it feeds into the broader Gender Agency and Control composite (DP1) that measures functional economic decision-making power. However, gender alone affects household income patterns: female-headed households often have different income composition (more livestock, trading, and food processing income) compared to male-headed households (more crop-dominant income).
**Related Data Points**: DP1 (Gender Agency & Control)

### age_group
**Type**: Categorical (Young, Early_Middle, Late_Middle, Senior)
**Description**: Derived age category that groups farmers into life stages. Young (0–20 years) represents pre-career farmers with minimal experience. Early_Middle (21–35) captures early-career farmers building their operations. Late_Middle (36–45) represents established mid-career farmers with solid experience. Senior (46+) includes the most experienced farmers with decades of agricultural knowledge. Note that these ML feature categories differ slightly from the DP2 scoring brackets (which use 25, 34, 46, 59 as boundaries).
**Derivation**: Binned from the farmer's raw age reported in Q10.
**Related Data Points**: DP2 (Age)

### family_size
**Type**: Numeric (integer, range 0–20)
**Description**: Total number of family members in the farmer's household, including the farmer themselves, spouse, children, elderly relatives, and other dependents. This feature directly impacts the income_per_family_member derived metric and serves as a proxy for household consumption burden. Larger families require more food, healthcare, and education spending, reducing the surplus available for loan repayment. However, larger families also provide more farm labor during peak seasons.
**Related Data Points**: DP3 (Number of Dependents)

### typeofhouse
**Type**: Categorical (permanent, semi-permanent, temporary)
**Description**: Classification of the farmer's dwelling as a proxy for accumulated wealth and asset stability. "Permanent" corresponds to modern housing (concrete/brick walls, corrugated iron roof, cement floor). "Semi-permanent" maps to traditional houses with multiple rooms. "Temporary" corresponds to traditional all-in-one dwellings. This feature captures long-term capital accumulation that income-based measures may miss.
**Related Data Points**: DP4 (Family Homestead Type)

---

## 2. Asset and Access Features

### asset_ownership
**Type**: Binary (1=yes, 0=no)
**Description**: Whether the farmer owns productive assets beyond basic hand tools. Productive assets include livestock, mechanized farm equipment, irrigation infrastructure, storage facilities, or other capital goods that enhance agricultural productivity. Asset ownership indicates capital accumulation and provides a financial buffer — assets can be liquidated in emergencies to maintain loan repayment.
**Related Data Points**: DP13 (Farm Tools), DP14 (Output Storage)

### water_reserve_access
**Type**: Binary (1=yes, 0=no)
**Description**: Whether the farmer has access to a reliable water reserve for supplemental irrigation. In rain-fed Ethiopian agriculture, water reserve access dramatically reduces climate risk by providing a buffer against drought and irregular rainfall. Farmers with irrigation or water reserves can sustain crop production through dry spells, maintaining yield and income even when neighboring rain-dependent farms fail. This is one of the most impactful binary features for climate resilience.
**Related Data Points**: DP18 (Economic Adaptive Capability)

### output_storage_type
**Type**: Categorical (warehouse, silo, none)
**Description**: The type of facility used to store harvested crops. "Warehouse" indicates access to formal storage (cooperative warehouse, commodity financing facility). "Silo" represents improved on-farm storage. "None" means the farmer lacks dedicated storage and must sell at harvest or store in suboptimal conditions. Storage type directly impacts post-harvest losses and the farmer's ability to time crop sales for optimal prices. Farmers with good storage can hold produce for months, selling when prices peak rather than at harvest when prices are lowest.
**Related Data Points**: DP14 (Output Storage & Control)

### decision_making_role
**Type**: Categorical (primary, secondary, joint)
**Description**: The farmer's role in household financial decisions. "Primary" means the farmer makes final decisions on major financial matters (crop sales, input purchases, loan applications). "Joint" indicates shared decision-making with a spouse or partner. "Secondary" means another household member controls major financial decisions. This feature is a key gender-intentional indicator — women who are primary or joint decision-makers demonstrate economic agency that predicts responsible loan management.
**Related Data Points**: DP1 (Gender Agency & Control)

### land_title
**Type**: Binary (1=yes, 0=no)
**Description**: Whether the farmer holds a formal land title certificate for their owned farmland. In Ethiopian law, land is state-owned with use rights granted to individuals, so a "land title" refers to a land use certificate. Formal title provides tenure security, collateral value for loans, and protection against land disputes. Joint titles (with spouse) are counted as "yes." This is one of the strongest binary predictors of creditworthiness due to its high reliability and direct collateral implications.
**Related Data Points**: DP11 (Land Title Availability)

### rented_farm_land
**Type**: Numeric (hectares, range 0–10)
**Description**: The area of farmland the farmer rents from others. Rented land expands the farmer's productive capacity but introduces ongoing rental costs (captured in DP7) and tenure uncertainty. Large rented areas may indicate ambitious farming operations but also higher fixed costs. The rental cost-to-revenue ratio matters — renting is financially viable only if the expected crop revenue exceeds the sum of rental and production costs.
**Related Data Points**: DP10 (Total Land Size Managed)

### own_farmland_size
**Type**: Numeric (hectares, range 0–10)
**Description**: The area of farmland the farmer personally owns (with formal or informal ownership). Owned land is the most stable productive asset — it does not carry rental costs and provides long-term security for agricultural investment. Larger owned farmland indicates greater wealth accumulation and production potential. Combined with land_title, this feature provides a comprehensive picture of the farmer's land asset position.
**Related Data Points**: DP10 (Total Land Size Managed)

### family_farmland_size
**Type**: Numeric (hectares, range 0–10)
**Description**: Total farmland controlled by the farmer's family, which the farmer has access to through family arrangements. Family land represents a middle ground between owned and rented land — it provides stable access without rental costs, but may be subject to family negotiations about use and output sharing. In Ethiopian households, family land arrangements often involve inter-generational transfers and shared use agreements.
**Related Data Points**: DP10 (Total Land Size Managed)

---

## 3. Institutional Features

### hasrusacco
**Type**: Binary (1=yes, 0=no)
**Description**: Whether the farmer is a member of a Rural Savings and Credit Cooperative Organization (RUSACCO). RUSACCOs are formal community-based financial institutions that provide savings accounts, small loans, and financial literacy training to rural members. Membership indicates financial engagement, savings discipline, and access to emergency credit. RUSACCO members typically have better financial management skills and stronger repayment track records due to the organization's accountability structures.
**Related Data Points**: DP18 (Economic Adaptive Capability), DP19 (Social Adaptive Capability)

### haslocaledir
**Type**: Binary (1=yes, 0=no)
**Description**: Whether the farmer is a member of a local Edir (traditional Ethiopian mutual aid society). Edirs are community organizations where members make regular contributions to a common fund, which is used to support members during funerals, illness, or other emergencies. Edir membership demonstrates social integration, regular financial commitment (similar to insurance premiums), and access to a community safety net. Members who face agricultural setbacks can draw on Edir support to bridge financial gaps without defaulting on other obligations.
**Related Data Points**: DP19 (Social & Human Resource Adaptive Capability)

### institutional_support_score
**Type**: Numeric (integer, range 0–4)
**Description**: A composite score summing four binary institutional engagement flags: (1) microfinance institution membership, (2) cooperative membership, (3) agricultural certification/training completion, and (4) community health insurance enrollment. Each "yes" adds 1 point. Higher scores indicate greater institutional engagement and access to formal support systems. Farmers with scores of 3–4 have diversified institutional connections that provide multiple layers of support — financial (microfinance), operational (cooperative), knowledge (certification), and health (insurance).
**Related Data Points**: DP18 (Economic Adaptive), DP19 (Social Adaptive), DP20 (Tech/Institutional)

---

## 4. Farm Operation Features

### primaryoccupation
**Type**: Categorical (farming, animal husbandry, mixed, other)
**Description**: The farmer's main livelihood activity. "Farming" indicates crop production is the primary income source. "Animal husbandry" means livestock is the dominant activity. "Mixed" indicates a diversified livelihood combining crops and livestock. "Other" captures non-agricultural primary occupations (trading, employment, services). Mixed farmers often have more stable income due to diversification, while pure crop farmers have more concentrated but potentially higher income from their primary activity.
**Related Data Points**: DP8 (Other Income)

### farmsizehectares
**Type**: Numeric (hectares, range 0–10)
**Description**: Total operated farm area in hectares, including both owned and rented land allocated to all crops. This is the primary land-based feature in the ML model and directly determines production scale. Farm size is the denominator in yield_per_hectare and input_intensity calculations, making it a foundational feature for productivity assessment. Ethiopian smallholders typically operate 0.5–3 hectares, with larger operations (5+ hectares) indicating commercial-scale farming.
**Related Data Points**: DP10 (Land Managed), DP12 (Land for Target Crop)

### farm_mechanization
**Type**: Categorical (manual, semi-mechanized, fully mechanized)
**Description**: The level of mechanization in the farmer's operations. "Manual" means all farm work is done with hand tools (hoe, sickle). "Semi-mechanized" indicates partial use of animal traction (oxen plowing) or rental access to tractors/threshers. "Fully mechanized" means the farmer owns or has reliable access to mechanized equipment for land preparation, planting, and/or harvesting. Higher mechanization increases productivity, reduces labor costs, and enables more timely operations (critical for rain-fed crops where planting windows are narrow).
**Related Data Points**: DP13 (Farm Tools), DP20 (Tech/Institutional)

### agriculture_experience
**Type**: Numeric (log-transformed)
**Description**: The farmer's agricultural experience in years, transformed using log1p (natural logarithm of 1 + years). The log transformation compresses the range so that the difference between 1 and 5 years of experience is weighted more heavily than the difference between 25 and 30 years, reflecting the diminishing marginal value of additional experience. Early years of farming involve steep learning curves, while experienced farmers have relatively stable skill levels.
**Related Data Points**: DP2 (Age, as a proxy)

### flaw
**Type**: Binary (1=yes, 0=no)
**Description**: Presence of observed land or crop defects, quality issues, or current damage. This includes soil degradation, erosion, waterlogging, current pest infestations, disease symptoms, or other observable problems that reduce productive capacity. The flaw indicator is assessed during field visits and represents current conditions rather than historical patterns. A "yes" flag indicates immediate productivity risk that may impact the current season's harvest and loan repayment capacity.
**Related Data Points**: DP16 (Climate Exposure, indirectly)

---

## 5. Input Features

### seedtype
**Type**: Categorical (improved, traditional, hybrid)
**Description**: The type of seed the farmer uses for their target crop. "Improved" seeds are government- or research-station-developed varieties with higher yield potential. "Traditional" or "local" seeds are farmer-saved varieties from previous harvests — lower yield potential but adapted to local conditions and zero cost. "Hybrid" seeds offer the highest yield potential but must be purchased each season (cannot be saved). Seed choice directly impacts expected yield and cost structure.
**Related Data Points**: DP5 (Cost of Target Crop), DP6 (Income from Target Crop)

### seedquintals
**Type**: Numeric (quintals, where 1 quintal = 100 kg)
**Description**: Quantity of seed used for the target crop. Seed quantity varies significantly by crop — teff requires only 10 kg/ha while potatoes need 1,200 kg/ha. The amount used relative to farm size indicates whether the farmer is following recommended seeding rates. Under-seeding may indicate financial constraints (cannot afford adequate seed), while over-seeding wastes resources. This feature feeds into the input_intensity ratio.
**Related Data Points**: DP5 (Cost of Target Crop)

### ureafertilizerquintals
**Type**: Numeric (quintals)
**Description**: Quantity of Urea fertilizer (46-0-0, high-nitrogen) applied to the target crop. Urea is the primary nitrogen source for cereal crops and is critical for vegetative growth. Application rates vary by crop — maize and tomato require 200 kg/ha while legumes (soybean, haricot bean, chickpea) need zero Urea due to biological nitrogen fixation. The quantity used indicates both the farmer's investment in productivity and their agronomic knowledge.
**Related Data Points**: DP5 (Cost of Target Crop)

### dapnpsfertilizerquintals
**Type**: Numeric (quintals)
**Description**: Quantity of DAP (Diammonium Phosphate, 18-46-0) or NPS (Nitrogen-Phosphorus-Sulfur blend) fertilizer applied. DAP/NPS provides phosphorus essential for root development and seed formation. Most crops require some DAP/NPS, with potatoes and maize needing the most (120 kg/ha) and sesame the least (0 kg/ha). Combined with Urea, this feature captures the farmer's total chemical fertilizer investment.
**Related Data Points**: DP5 (Cost of Target Crop)

---

## 6. Yield Features

### expectedyieldquintals
**Type**: Numeric (quintals)
**Description**: The farmer's expected total harvest yield from their target crop, measured in quintals. This represents the gross production before accounting for household consumption or post-harvest losses. Expected yield varies dramatically by crop — from 8 qtl/ha for sesame to 200 qtl/ha for potatoes. The relationship between expected yield and farm size (yield_per_hectare) is a key productivity indicator. Higher-than-average yields for a given crop suggest good management practices.
**Related Data Points**: DP6 (Income from Target Crop)

### saleableyieldquintals
**Type**: Numeric (quintals)
**Description**: The portion of the harvest the farmer intends to sell at market, measured in quintals. The difference between expected yield and saleable yield represents household consumption, seed saving for next season, animal feed, and anticipated post-harvest losses. Marketable percentage ranges from 70% (teff, due to high household consumption) to 95% (tomato, coffee). Higher saleable yield means more revenue available for loan repayment.
**Related Data Points**: DP6 (Income from Target Crop)

### yield_per_hectare
**Type**: Numeric (quintals per hectare)
**Description**: Expected yield divided by total operated farm area — the key productivity indicator in the feature set. Yield per hectare normalizes production across different farm sizes, enabling comparison between a 0.5 ha farmer and a 5 ha farmer. Higher yield per hectare indicates better management practices (appropriate inputs, timing, pest control) and/or better growing conditions (soil quality, rainfall). This feature is one of the strongest predictors of net income and, consequently, repayment capacity.
**Related Data Points**: DP6 (Income from Target Crop), DP12 (Land for Target Crop)

---

## 7. Income and Financial Features

### total_estimated_income
**Type**: Numeric (ETB)
**Description**: The sum of all income sources: primary target crop revenue, other crop revenues, livestock income (dairy, animal sales), off-farm income (salary, remittances, rental), and trading/processing income. This comprehensive income measure captures the farmer's total cash-generating capacity. Higher total income provides a larger base from which loan repayments can be made.
**Related Data Points**: DP6 (Income Target Crop), DP8 (Other Income)

### income_per_family_member
**Type**: Numeric (ETB per person)
**Description**: Total estimated income divided by family size — a per-capita welfare indicator. This derived feature captures the actual economic capacity available to each household member, accounting for household size. A farmer earning 50,000 ETB with 3 family members has fundamentally different welfare (16,667 ETB/person) than one earning the same amount with 10 family members (5,000 ETB/person). Higher income per family member indicates better household welfare and more capacity for debt service.
**Related Data Points**: DP3 (Dependents), DP6 (Income), DP8 (Other Income)

### total_estimated_cost
**Type**: Numeric (ETB)
**Description**: The sum of all production expenses (target crop inputs, other crop inputs) and household costs (student support, land rent, veterinary services, living expenses). This comprehensive cost measure captures the farmer's total financial obligations. Lower total costs relative to income indicate better profitability and more capacity for loan repayment.
**Related Data Points**: DP5 (Cost Target Crop), DP7 (Other Costs)

### net_income
**Type**: Numeric (ETB)
**Description**: Total estimated income minus total estimated cost — the primary creditworthiness indicator in the feature set. Net income represents the actual surplus available to the farmer after all production and household expenses. This is the fundamental measure of a farmer's ability to repay a loan: if net income is positive and exceeds the loan repayment amount, the farmer has the capacity to service the debt. Net income is influenced by crop choice, farm size, input efficiency, income diversification, and household expense management.
**Related Data Points**: DP5 (Cost), DP6 (Revenue), DP7 (Other Costs), DP8 (Other Income)

---

## 8. Target Variable

### decision
**Type**: Categorical (Eligible, Review, Not Eligible)
**Description**: The target label for ML model training and prediction. This three-class outcome represents the final credit eligibility decision:
- **Eligible**: The farmer meets creditworthiness criteria across multiple dimensions and is recommended for loan approval.
- **Review**: The farmer presents a borderline profile requiring manual assessment by a loan officer — some indicators are favorable while others raise concerns.
- **Not Eligible**: The farmer does not currently meet creditworthiness requirements and is not recommended for loan approval at this time.

The decision is derived from the composite of all 20 Data Point scores, percentile ranking, and the ML model's learned patterns from historical data.

---

## Input Intensity: A Special Derived Feature

### input_intensity
**Type**: Numeric (ratio)
**Description**: A derived ratio calculated as `(seedquintals + ureafertilizerquintals + dapnpsfertilizerquintals) / farmsizehectares`. This proxy for farming intensity measures how many quintals of total inputs the farmer applies per hectare of land. Higher input intensity suggests more intensive farming practices with greater investment per unit of land. Very high input intensity may indicate over-application (waste), while very low intensity may indicate under-investment (suboptimal yields). The optimal range depends on crop type and local agronomic recommendations.
**Related Data Points**: DP5 (Cost of Target Crop), DP12 (Land for Target Crop)
