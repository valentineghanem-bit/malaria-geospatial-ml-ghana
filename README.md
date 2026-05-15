# Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 260-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and WASH Determinants

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana  
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)  
**Reporting standard:** STROBE  
**Date:** April 2026

> Ghanem VG. *Geospatial clustering and machine learning prediction of malaria burden at 260-district resolution in Ghana.* 2026.

**Note on licensing:** Code is released under MIT. Data outputs and figures are released under CC BY 4.0.

---

## Overview

This study maps malaria burden across Ghana's 260 districts at unprecedented subnational resolution, integrating insecticide-treated net (ITN) coverage and WASH determinants. Spatial clustering identifies high-priority hotspot districts, and ensemble machine learning (XGBoost, Random Forest, CART, Logistic Regression) with Leave-One-District-Out cross-validation produces calibrated district-level malaria risk predictions. SHAP TreeExplainer identifies ITN coverage as the dominant modifiable predictor.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Global Moran's I (malaria incidence) | 0.672 (z=14.38, p<0.001) |
| Bivariate LISA High-High clusters (ITN deficit × malaria) | 36 |
| Priority-1 Gi* hotspot districts (99.9% CI) | 38 |
| All hotspot districts (all tiers) | 51 |
| XGBoost AUC-ROC | 0.923 (95% CI: 0.907–0.939), LODO-CV |
| Top SHAP predictor | ITN coverage (\|SHAP\|=0.41) |

---

## Repository Structure

```
malaria-geospatial-ml-ghana/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── spatial/       (01–05)
│   ├── ml/            (06–10)
│   └── figures/
├── dashboard/
│   └── Ghana_Malaria_260District_Dashboard.html
├── poster/
├── tests/
├── docs/
├── requirements.txt
├── Dockerfile
└── CITATION.cff
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana.git
cd malaria-geospatial-ml-ghana
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
# Spatial analysis (01–05)
python scripts/spatial/01_spatial_weights.py
python scripts/spatial/02_global_moran.py
python scripts/spatial/03_lisa.py
python scripts/spatial/04_getis_ord.py
python scripts/spatial/05_bivariate_lisa.py

# ML pipeline (06–10)
python scripts/ml/06_feature_engineering.py
python scripts/ml/07_xgboost_lodo.py
python scripts/ml/08_random_forest.py
python scripts/ml/09_ensemble.py
python scripts/ml/10_shap.py

# Figures
python scripts/figures/generate_figures.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Open the interactive dashboard

Open `dashboard/Ghana_Malaria_260District_Dashboard.html` in any modern browser. No server required.

---

## Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| WHO Global Health Observatory | Malaria indicators | 2001–2023 | who.int/data/gho (open) |
| Ghana DHS Programme | Subnational parasitaemia, ITN, WASH, U5MR | 2003–2022 | dhsprogram.com (DUA required) |
| Ghana DHIMS2 | Routine malaria cases | 2018–2022 | Ghana Health Service |
| Ghana Statistical Service | District boundaries | 2021 | statsghana.gov.gh |
| WorldPop | Population surface | 2020 | worldpop.org (open) |

---

## Methods Summary

| Method | Tool | Purpose |
|--------|------|---------|
| Global Moran's I (KNN k=8) | esda / libpysal | Spatial autocorrelation of malaria incidence |
| Bivariate LISA (Rook contiguity) | esda | ITN deficit × malaria co-clustering |
| Getis-Ord Gi* | esda | Hotspot tiering (99.9%, 99%, 95% CI) |
| XGBoost LODO-CV | xgboost | Risk prediction with spatial cross-validation |
| Random Forest | scikit-learn | Ensemble predictor importance |
| CART | scikit-learn | Interpretable decision rules |
| Logistic Regression | scikit-learn | Baseline classification |
| SHAP TreeExplainer | shap | Feature attribution and interpretability |

---

## Reproducibility

- Random seed: 42 throughout  
- Reporting: STROBE  
- All random seeds set explicitly (`random_state=42`)  
- DHS data accessed under signed Data Use Agreement (ICF International)

---

## Ethical Statement

This study used exclusively secondary data. No primary data collection from human participants was conducted. Ghana DHS data were accessed under a signed Data Use Agreement with ICF International. Ethical review was therefore not required for this analysis.

---

## Citation

```bibtex
@misc{ghanem2026malaria,
  author = {Ghanem, Valentine Golden},
  title  = {Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 260-District Resolution in Ghana},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details. Data outputs and figures: CC BY 4.0.

---

## Contact

Valentine Golden Ghanem  
Ghana COCOBOD Cocoa Clinic, Accra, Ghana  
valentineghanem@gmail.com  
ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
