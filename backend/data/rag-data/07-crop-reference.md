# Crop Reference Data (CropRef)

## Overview

The Crop Reference Data table (CropRef) is the external data engine that standardizes cost and revenue projections for all supported crops. It eliminates subjectivity from financial calculations by providing validated per-hectare input requirements, costs, yields, and market prices. The CropRef data is the foundation for DP5 (Cost of Target Crop) and DP6 (Income from Target Crop) calculations — the two highest-weighted data points in the scoring framework (15 points each).

## How CropRef Is Used in Credit Scoring

1. The farmer selects their target crop (Q25) and reports their cultivation area in hectares (Q26)
2. The system looks up the crop in CropRef to find standardized per-hectare costs and revenue
3. **DP5 calculation**: `Total_Cost = CropRef_Cost_Per_Ha × Farmer_Area_Ha`
4. **DP6 calculation**: `Total_Revenue = CropRef_Revenue_Per_Ha × Farmer_Area_Ha`
5. Both values are quartile-scored against the farmer population

This lookup-based approach ensures that farmers cannot overstate revenue or understate costs, which is why DP5 and DP6 both have Reliability = 1.0 (High).

## Complete Crop Reference Table (2025 Data)

| Crop | Seed Qty/Ha | Seed Price (ETB) | Urea Qty/Ha | Urea Price/Qtl | DAP Qty/Ha | DAP Price/Qtl | Chemical Type | Chemical Price (ETB) | Total Cost/Ha (ETB) | Yield/Ha (Qtl) | Marketable % | Price/Qtl (ETB) | Revenue/Ha (ETB) |
|------|------------|------------------|------------|----------------|-----------|--------------|--------------|---------------------|-------------------|----------------|-------------|----------------|-----------------|
| Maize | 15 kg | 130 | 200 kg | 8 | 120 kg | 9 | Insecticide C | 550 | 5,015 | 30 | 85% | 1,600 | 40,800 |
| Wheat | 150 kg | 11 | 140 kg | 8 | 90 kg | 10 | Fungicide B | 600 | 3,390 | 22 | 75% | 1,700 | 28,050 |
| Malt Barley | 120 kg | 12 | 150 kg | 8 | 100 kg | 9 | Herbicide A | 500 | 3,940 | 25 | 80% | 1,800 | 36,000 |
| Teff | 10 kg | 100 | 50 kg | 8 | 40 kg | 9 | — | — | 1,760 | 12 | 70% | 2,000 | 16,800 |
| Sorghum | 10 kg | 110 | 120 kg | 8 | 80 kg | 9 | Herbicide A | 500 | 3,030 | 18 | 75% | 1,500 | 20,250 |
| Millet | 8 kg | 125 | 60 kg | 8 | 50 kg | 9 | Herbicide A | 500 | 1,880 | 10 | 80% | 1,400 | 11,200 |
| Soybean | 40 kg | 35 | 0 | — | 70 kg | 9 | Herbicide A | 500 | 2,000 | 18 | 90% | 2,200 | 35,640 |
| Haricot Bean | 50 kg | 32 | 0 | — | 60 kg | 9 | Herbicide A | 550 | 1,505 | 12 | 90% | 2,600 | 28,080 |
| Chickpea | 60 kg | 25 | 0 | — | 80 kg | 9 | Herbicide A | 550 | 1,840 | 15 | 85% | 2,400 | 30,600 |
| Tomato | 15k seedlings | 0.15/each | 200 kg | 9.5 | 0 | — | Pesticide Mix | 700 | 4,850 | 150 | 95% | 400 | 57,000 |
| Onion | 600 sets | 4/each | 180 kg | 8 | 100 kg | 9 | Fungicide B | 600 | 5,280 | 120 | 90% | 350 | 37,800 |
| Potatoes | 1,200 kg | 2/kg | 160 kg | 8 | 120 kg | 9 | Insecticide C | 550 | 5,200 | 200 | 85% | 300 | 51,000 |
| Sesame | 4 kg | 100 | 100 kg | 8 | 0 | — | Herbicide A | 600 | 1,560 | 8 | 90% | 4,000 | 28,800 |
| Coffee | 2k seedlings | 2/each | 0 | — | 0 | — | — | — | 4,000 | 35 | 95% | 3,500 | 116,375 |

## Individual Crop Profiles

### Maize

- **Cost per hectare**: 5,015 ETB
- **Revenue per hectare**: 40,800 ETB
- **Profit margin per hectare**: 35,785 ETB
- **Input profile**: High fertilizer needs (200 kg Urea + 120 kg DAP), moderate seed cost, insecticide protection required
- **Key characteristics**: Widely cultivated staple crop across Ethiopian lowlands and mid-altitudes. High yield potential (30 qtl/ha) with 85% marketable. Sensitive to moisture stress during flowering — drought at tasseling can reduce yields by 50% or more.
- **Credit implications**: Good profit margins but high input costs create financial exposure. Suitable for farmers with established capital or credit access.

### Wheat

- **Cost per hectare**: 3,390 ETB
- **Revenue per hectare**: 28,050 ETB
- **Profit margin per hectare**: 24,660 ETB
- **Input profile**: High seed rate (150 kg/ha), moderate fertilizer, fungicide protection needed
- **Key characteristics**: Major highland crop with strong domestic demand. Lower marketable percentage (75%) due to household consumption retention. Reliable market with relatively stable pricing.
- **Credit implications**: Moderate costs and steady returns make wheat a lower-risk crop for lending. Well-suited for risk-averse lending strategies.

### Malt Barley

- **Cost per hectare**: 3,940 ETB
- **Revenue per hectare**: 36,000 ETB
- **Profit margin per hectare**: 32,060 ETB
- **Input profile**: Balanced across all input categories with herbicide protection
- **Key characteristics**: Premium crop with higher per-quintal price (1,800 ETB) driven by brewery demand. 80% marketable yield. Highland-adapted crop requiring reliable rainfall timing.
- **Credit implications**: Strong profit margins and premium market linkage (breweries) provide reliable income streams. Good candidate for credit-financed production.

### Teff

- **Cost per hectare**: 1,760 ETB
- **Revenue per hectare**: 16,800 ETB
- **Profit margin per hectare**: 15,040 ETB
- **Input profile**: Very low inputs — minimal seed (10 kg), low fertilizer (50 kg Urea, 40 kg DAP), no chemicals
- **Key characteristics**: Ethiopia's signature grain, staple for injera. Lowest input requirements of all cereals. Only 70% marketable due to high household consumption. Premium price (2,000 ETB/qtl) but low yield (12 qtl/ha).
- **Credit implications**: Very low cost exposure makes teff low-risk for lending, but absolute revenue is also lower. Best for smaller loan amounts.

### Sorghum

- **Cost per hectare**: 3,030 ETB
- **Revenue per hectare**: 20,250 ETB
- **Profit margin per hectare**: 17,220 ETB
- **Input profile**: Moderate across all categories with herbicide
- **Key characteristics**: Drought-tolerant crop well-adapted to lowland and semi-arid conditions. Lower yield (18 qtl/ha) and price (1,500 ETB/qtl) but highly resilient. 75% marketable.
- **Credit implications**: Excellent for climate-risk-adjusted lending. Low default risk due to drought tolerance, though absolute returns are moderate.

### Millet

- **Cost per hectare**: 1,880 ETB
- **Revenue per hectare**: 11,200 ETB
- **Profit margin per hectare**: 9,320 ETB
- **Input profile**: Low across all categories
- **Key characteristics**: Most drought-resistant cereal. Lowest yield (10 qtl/ha) and lowest revenue per hectare. 80% marketable. Grown primarily in marginal areas.
- **Credit implications**: Lowest absolute revenue limits loan size, but very low default risk due to climate resilience. Suitable for micro-credit in drought-prone zones.

### Soybean

- **Cost per hectare**: 2,000 ETB
- **Revenue per hectare**: 35,640 ETB
- **Profit margin per hectare**: 33,640 ETB
- **Input profile**: No Urea needed (nitrogen-fixing legume), moderate DAP and herbicide
- **Key characteristics**: High-value legume with excellent profit margins. No nitrogen fertilizer required (biological fixation). 90% marketable. Growing domestic and export demand.
- **Credit implications**: Outstanding cost-to-revenue ratio (1:17.8). Very low input costs combined with high revenue make soybean one of the best crops for credit-financed production.

### Haricot Bean

- **Cost per hectare**: 1,505 ETB
- **Revenue per hectare**: 28,080 ETB
- **Profit margin per hectare**: 26,575 ETB
- **Input profile**: Lowest total cost of any crop; no Urea
- **Key characteristics**: Lowest-cost crop to produce. High marketable percentage (90%) and strong export demand. Good price (2,600 ETB/qtl).
- **Credit implications**: Exceptional cost-to-revenue ratio. Minimal financial exposure with strong return. Ideal for first-time borrowers or risk-minimizing strategies.

### Chickpea

- **Cost per hectare**: 1,840 ETB
- **Revenue per hectare**: 30,600 ETB
- **Profit margin per hectare**: 28,760 ETB
- **Input profile**: Low cost; no Urea (nitrogen fixer), moderate DAP
- **Key characteristics**: Another nitrogen-fixing legume with strong domestic and export markets. High price (2,400 ETB/qtl), 85% marketable.
- **Credit implications**: Strong profit margins with low input cost. Legume crop rotation benefits also improve subsequent cereal yields, creating multi-season value.

### Tomato

- **Cost per hectare**: 4,850 ETB
- **Revenue per hectare**: 57,000 ETB
- **Profit margin per hectare**: 52,150 ETB
- **Input profile**: High seedling count, heavy Urea needs (200 kg at 9.5 ETB/qtl), pesticide mix required
- **Key characteristics**: Highest-yield crop (150 qtl/ha) with 95% marketable. Water-intensive and highly perishable. Requires irrigation or reliable rainfall. Subject to significant price volatility.
- **Credit implications**: Very high revenue potential but high input costs and climate sensitivity create substantial risk. Best for experienced farmers with irrigation access.

### Onion

- **Cost per hectare**: 5,280 ETB
- **Revenue per hectare**: 37,800 ETB
- **Profit margin per hectare**: 32,520 ETB
- **Input profile**: High costs across all categories including fungicide
- **Key characteristics**: High-value horticultural crop. 120 qtl/ha yield with 90% marketable. Requires careful moisture management — sensitive to both drought and excess water.
- **Credit implications**: Good profit margins but highest total input cost. Climate sensitivity and storage requirements add risk. Suitable for farmers with demonstrated horticultural experience.

### Potatoes

- **Cost per hectare**: 5,200 ETB
- **Revenue per hectare**: 51,000 ETB
- **Profit margin per hectare**: 45,800 ETB
- **Input profile**: High seed tuber cost (1,200 kg at 2 ETB/kg), heavy fertilizer, insecticide
- **Key characteristics**: Highest yield per hectare (200 qtl) among all crops. 85% marketable. Requires cool highland conditions with consistent moisture. Bulky harvest requires good storage.
- **Credit implications**: Strong absolute revenue with good margins. High input costs require significant upfront capital. Well-suited for highland farmers with storage access.

### Sesame

- **Cost per hectare**: 1,560 ETB
- **Revenue per hectare**: 28,800 ETB
- **Profit margin per hectare**: 27,240 ETB
- **Input profile**: Minimal — low seed cost, moderate Urea only, herbicide
- **Key characteristics**: Premium export crop with the highest per-quintal price (4,000 ETB) among annual crops. Low yield (8 qtl/ha) but excellent value. 90% marketable. Heat-tolerant.
- **Credit implications**: Excellent cost-to-revenue ratio with export market stability. Low input requirements minimize risk. Strong choice for credit in lowland/semi-arid zones.

### Coffee

- **Cost per hectare**: 4,000 ETB
- **Revenue per hectare**: 116,375 ETB
- **Profit margin per hectare**: 112,375 ETB
- **Input profile**: Seedling establishment cost; no fertilizer or chemicals in reference data (organic/traditional production)
- **Key characteristics**: Ethiopia's premier export crop with by far the highest revenue per hectare. Perennial crop (not replanted annually). 95% marketable. 35 qtl/ha yield at 3,500 ETB/qtl.
- **Credit implications**: Highest absolute profit margin of any crop. Long-term asset (perennial). Strong export demand. However, requires 3-5 years to reach full production, making it unsuitable for short-term loan cycles unless the trees are already mature.

## Crop Groupings by Risk Profile

### Low Risk (Low Cost, Moderate-High Revenue)
- Haricot Bean, Soybean, Chickpea, Sesame
- Characterized by low input costs (especially no Urea for legumes), strong profit margins, and climate resilience
- Best for conservative lending or first-time borrowers

### Moderate Risk (Moderate Cost, Good Revenue)
- Wheat, Malt Barley, Teff, Sorghum, Millet
- Staple cereals with predictable demand and moderate input requirements
- Solid candidates for standard agricultural credit

### Higher Risk (High Cost, High Revenue)
- Maize, Tomato, Onion, Potatoes
- High input costs and climate/water sensitivity create greater exposure
- Suitable for experienced farmers with demonstrated capacity

### Special Category (Perennial)
- Coffee: Highest revenue but requires mature trees; long payback period
- Best financed through multi-year credit products
