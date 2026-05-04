# Ghana Malaria 260-District Geospatial & ML Analysis

**Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 260-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and Water, Sanitation, and Hygiene Determinants**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana 
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220) 
**Target journal:** *Malaria Journal* / *PLOS Global Public Health* / *BMC Public Health* 
**Reporting standard:** STROBE (observational ecological study) 
**Date:** April 2026

---

## Overview

This repository contains all code, data, and analytical outputs for a comprehensive geospatial and machine learning analysis of malaria burden across all 260 districts of Ghana. The study integrates ITN coverage, WASH indicators, parasitaemia prevalence, under-five mortality, and malaria incidence data to:

1. Characterise spatial autocorrelation (Global Moran's I, k=8 KNN weights)
2. Identify co-clustering of ITN deficits and malaria incidence (Bivariate LISA, Rook contiguity)
3. Delineate statistically significant hotspot/coldspot districts (Getis-Ord Gi*)
4. Develop and validate ML risk scores (XGBoost, Random Forest, CART, Logistic Regression)
5. Interpret dominant determinants using SHAP TreeExplainer
6. Produce a deployable interactive dashboard for the Ghana National Malaria Elimination Programme

**Key findings:**
- Global Moran's I (malaria incidence) = 0.672 (z=14.38, p<0.001)
- 36 Bivariate LISA High-High clusters (ITN deficit × high malaria incidence co-clustering)
- 38 Priority-1 Gi* hotspot districts (99.9% CI); 51 districts across all hotspot significance tiers
- XGBoost AUC-ROC: 0.923 (95% CI: 0.907–0.939) — leave-one-district-out spatial CV
- SHAP top-3: ITN coverage (|SHAP|=0.41) > Parasitaemia (0.38) > Water access (0.27)

---

## Repository Structure

```
Ghana_Malaria_260District_Repo/
├── data/
│ ├── raw/ # Original source datasets (read-only)
│ │ ├── malaria_indicators_gha.csv (WHO GHO 2001–2023)
│ │ ├── malaria-parasitemia_subnational.csv (Ghana DHS 2014)
│ │ ├── insecticide-treated-nets.csv (Ghana DHS series)
│ │ ├── water_subnational.csv (Ghana DHS / JMP)
│ │ ├── rbm_subnational_gha.csv (U5MR, RBM)
│ │ └── Ghana_New_260_District.geojson (GSS 2021)
│ └── processed/
│ └── Ghana_Malaria_260District_MasterDataset.csv (Master output)
├── scripts/
│ ├── spatial/
│ │ ├── 01_spatial_weights.py (KNN + Rook contiguity matrices)
│ │ ├── 02_global_morans.py (Global Moran's I + permutation test)
│ │ ├── 03_bivariate_lisa.py (Bivariate LISA, ITN × incidence)
│ │ ├── 04_getis_ord.py (Gi* hotspot/coldspot delineation)
│ │ └── 05_spatial_regression.py (SLM/SEM via robust LM test)
│ ├── ml/
│ │ ├── 06_feature_engineering.py (Feature matrix construction)
│ │ ├── 07_xgboost_model.py (XGBoost + LODO-CV + calibration)
│ │ ├── 08_random_forest.py (Random Forest + LODO-CV)
│ │ ├── 09_cart_logistic.py (CART + Logistic regression)
│ │ └── 10_shap_interpretability.py (SHAP beeswarm + waterfall + dependence)
│ └── figures/
│ └── generate_figures.py (All 6 publication figures at 300 DPI)
├── notebooks/
│ └── 00_exploratory_analysis.ipynb (Data profiling + Table 1)
├── dashboard/
│ └── Ghana_Malaria_260District_Dashboard.html (Self-contained interactive dashboard)
├── tests/
│ ├── test_spatial.py (Unit tests: Moran's I, LISA cluster counts)
│ └── test_ml.py (Unit tests: AUC bounds, SHAP stability)
├── docs/
│ └── data_dictionary.md (Variable definitions, units, sources)
├── requirements.txt
├── CITATION.cff
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/valentineghanem/ghana-malaria-260district.git
cd ghana-malaria-260district

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (pinned)
pip install -r requirements.txt
```

---

## Reproducibility — Full Pipeline Execution

Run the complete analysis pipeline in order:

```bash
# Step 1: Spatial analysis
python scripts/spatial/01_spatial_weights.py
python scripts/spatial/02_global_morans.py
python scripts/spatial/03_bivariate_lisa.py
python scripts/spatial/04_getis_ord.py
python scripts/spatial/05_spatial_regression.py

# Step 2: Machine learning
python scripts/ml/06_feature_engineering.py
python scripts/ml/07_xgboost_model.py
python scripts/ml/08_random_forest.py
python scripts/ml/09_cart_logistic.py
python scripts/ml/10_shap_interpretability.py

# Step 3: Figures
python scripts/figures/generate_figures.py

# Step 4: Run tests
pytest tests/ -v
```

All outputs written to `data/processed/` and `scripts/figures/`.

---

## Data Sources

| Dataset | Source | Year | Access |
|---------|--------|------|--------|
| National malaria indicators (incidence, mortality) | WHO Global Health Observatory | 2001–2023 | Open: https://www.who.int/data/gho |
| Subnational parasitaemia, ITN, WASH, U5MR | Ghana DHS Programme (ICF International) | 2003–2022 | Registration: https://dhsprogram.com |
| District-level routine malaria cases | Ghana DHIMS2 | 2018–2022 | GHS application |
| District boundary polygons (260 districts) | Ghana Statistical Service | 2021 | https://www.statsghana.gov.gh |
| Population surface (spatial disaggregation) | WorldPop 2020 Ghana 100m grid | 2020 | https://www.worldpop.org |

---

## Key Dependencies

See `requirements.txt` for pinned versions. Core packages:

- `geopandas==0.14.3` — spatial data processing
- `libpysal==4.9.2` — spatial weights construction
- `esda==2.6.0` — Moran's I, LISA, Gi*
- `spreg==1.7.2` — spatial regression (SLM/SEM)
- `xgboost==2.0.3` — gradient boosted trees
- `scikit-learn==1.5.0` — ML pipeline, LODO-CV
- `shap==0.45.1` — SHAP TreeExplainer
- `matplotlib==3.8.2` + `seaborn==0.13.2` — figures
- `plotly==5.18.0` — dashboard
- `pandas==2.1.4` + `numpy==1.26.3`
- `scipy==1.12.0`

---

## Ethical Statement

This study used exclusively de-identified, publicly available secondary datasets. No primary data collection from human participants was conducted. Ethical review was therefore not required under Ghana Health Service Ethics Review Committee policy for secondary data analyses. DHS data accessed under signed Data Use Agreement (ICF International).

---

## Citation

If you use this repository, please cite:

```
Ghanem VG. Geospatial clustering and machine learning prediction of malaria burden at
260-district resolution in Ghana: integrating insecticide-treated net coverage and WASH
determinants. [Journal]. 2026. DOI: [pending]
```

See `CITATION.cff` for full machine-readable citation.

---

## License

Code: [MIT License](LICENSE) 
Data outputs and figures: [CC BY 4.0](https://creativecommons.org/