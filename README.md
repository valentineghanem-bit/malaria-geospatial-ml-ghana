# Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 261-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and WASH Determinants

[![CI](https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**Reporting standard:** STROBE
**Date:** April 2026
**Status:** Manuscript in preparation

---

## 1. Abstract

This study maps malaria burden across Ghana's 261 districts at subnational resolution, integrating insecticide-treated net (ITN) coverage and WASH determinants. Spatial clustering analysis identifies high-priority hotspot districts, while bivariate LISA quantifies ITN-deficit co-clustering with malaria incidence. An ensemble machine learning pipeline (XGBoost, Random Forest, CART, Logistic Regression) with Leave-One-District-Out (LODO) cross-validation produces calibrated district-level malaria risk predictions. SHAP TreeExplainer identifies ITN coverage as the dominant modifiable predictor, while water access and open defecation emerge as key structural cofactors.

---

## 2. Research Question & Aims

- **Primary:** Quantify the subnational distribution of malaria burden and identify priority hotspot districts across Ghana's 261 districts.
- **Secondary:** (a) Detect ITN-deficit × malaria co-clusters using bivariate LISA; (b) build a LODO-CV ensemble ML pipeline for district-level risk prediction; (c) interpret model drivers using SHAP TreeExplainer; (d) tier hotspot districts by significance for programme prioritisation.

---

## 3. Methods Summary

| Method | Tool | Purpose |
|--------|------|---------|
| Global Moran's I (KNN k=8) | esda / libpysal | Spatial autocorrelation of malaria incidence |
| Bivariate LISA (Rook contiguity) | esda | ITN deficit × malaria co-clustering |
| Getis-Ord Gi* | esda | Hotspot tiering (99.9%, 99%, 95% CI) |
| XGBoost (LODO-CV) | xgboost | Risk prediction with spatial cross-validation |
| Random Forest | scikit-learn | Ensemble predictor importance |
| CART | scikit-learn | Interpretable decision rules |
| Logistic Regression | scikit-learn | Baseline classification |
| SHAP TreeExplainer | shap | Feature attribution and interpretability |
| Spatial regression diagnostics | spdep / spatialreg (R) | OLS / SLM / SEM model selection |

---

## 4. Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| WHO Global Health Observatory | Malaria indicators | 2001–2023 | [who.int/data/gho](https://www.who.int/data/gho) (open) |
| Ghana DHS Programme | Subnational parasitaemia, ITN, WASH, U5MR | 2003–2022 | [dhsprogram.com](https://dhsprogram.com) (registration) |
| Ghana DHIMS2 | Routine malaria cases | 2018–2022 | Ghana Health Service |
| Ghana Statistical Service | District boundaries (261 districts) | 2021 | [statsghana.gov.gh](https://statsghana.gov.gh) |
| WorldPop | Population surface | 2020 | [worldpop.org](https://worldpop.org) (open) |

> DHS data accessed under signed Data Use Agreement (ICF International). No individual participant data redistributed.

---

## 5. Key Findings

| Metric | Value |
|--------|-------|
| Global Moran's I (malaria incidence) | 0.845 (p < 0.001) |
| LISA High-High clusters | 55 districts |
| LISA Low-Low clusters | 42 districts |
| Gi* hotspots (≥95% CI, all tiers) | 40 districts (22 at 99.9%, 16 at 99%, 2 at 95%) |
| Bivariate Moran's I (ITN deficit × malaria) | −0.471 (protective direction) |
| RandomForest 5-fold CV AUC | 0.725 |
| Top SHAP predictor | ITN coverage |
| Districts analysed | 261 (Guan District added 2026-05) |

---

## 6. Repository Structure

```
malaria-geospatial-ml-ghana/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── spatial/                    # 01–05 spatial analysis scripts
│   ├── ml/                         # 06–10 machine learning scripts
│   ├── figures/
│   │   └── generate_figures.py     # 300 DPI publication figures
│   ├── spatial_utils.py            # Reusable spatial analysis utilities
│   └── spatial_diagnostics.R       # R: spatial autocorrelation diagnostics
├── app.py                          # Plotly Dash interactive application
├── analysis.R                      # R: spatial regression + NB GLM
├── dashboard/
│   └── Ghana_Malaria_260District_Dashboard.html
├── poster/
│   └── Ghana_Malaria_260District_Poster.html
├── tests/
│   ├── test_ml.py
│   └── test_spatial.py
├── docs/
├── requirements.txt
├── Dockerfile
└── CITATION.cff
```

---

## 7. Reproducibility

### 7.1 Requirements

- Python 3.12 (pinned in `requirements.txt`)
- R 4.3+ with packages: spdep, spatialreg, MASS, dplyr (see `analysis.R` header)
- Random seed: 42 throughout
- Estimated runtime: ~8–12 minutes on a standard laptop
- Tested on: Ubuntu 22.04 / macOS 14 / Windows 11 (CI: GitHub Actions)

### 7.2 Clone & install

```bash
git clone https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana.git
cd malaria-geospatial-ml-ghana
pip install -r requirements.txt
```

### 7.3 Run the analytical pipeline

```bash
# Spatial analysis
python scripts/spatial/01_build_weights.py
python scripts/spatial/02_moran.py
python scripts/spatial/03_lisa.py
python scripts/spatial/04_hotspots.py
python scripts/spatial/05_bivariate.py
# Machine learning
python scripts/ml/06_features.py
python scripts/ml/07_xgboost.py
python scripts/ml/08_ensemble.py
python scripts/ml/09_shap.py
python scripts/ml/10_figures.py
```

### 7.4 Run the test suite

```bash
pytest tests/ -v
```

### 7.5 Launch the interactive Dash application

```bash
python app.py
# Visit http://127.0.0.1:8050
```

### 7.6 Open the static HTML dashboard

```bash
# macOS
open dashboard/Ghana_Malaria_260District_Dashboard.html
# Windows
start dashboard/Ghana_Malaria_260District_Dashboard.html
# Linux
xdg-open dashboard/Ghana_Malaria_260District_Dashboard.html
```

---

## 8. Outputs

| Output | Description |
|--------|-------------|
| `data/processed/` | Master CSV, spatial weights, LISA results, SHAP values |
| `figures/` | Publication-quality PNG figures (300 DPI) |
| `dashboard/` | Self-contained interactive HTML dashboard |
| `poster/` | A0 conference poster (HTML, print-ready) |

## 8a. Downloadable Artefacts (HTML)

Both the interactive dashboard and the conference poster are committed as self-contained HTML files — no server, no build step required.

| Artefact | View on GitHub | Live preview | Direct download (raw HTML) |
|----------|---------------|--------------|---------------------------|
| Interactive dashboard | [View](https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana/blob/main/dashboard/Ghana_Malaria_260District_Dashboard.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana/blob/main/dashboard/Ghana_Malaria_260District_Dashboard.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/malaria-geospatial-ml-ghana/main/dashboard/Ghana_Malaria_260District_Dashboard.html) |
| Conference poster | [View](https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana/blob/main/poster/Ghana_Malaria_260District_Poster.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana/blob/main/poster/Ghana_Malaria_260District_Poster.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/malaria-geospatial-ml-ghana/main/poster/Ghana_Malaria_260District_Poster.html) |

> **Tip:** The dashboard works fully offline once downloaded. The poster is print-ready at A0 (841 × 1189 mm).

---

## 9. Reporting Standard

This study follows the **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology) reporting guideline for observational ecological studies.

---

## 10. Ethical Statement

This study analyses publicly released aggregate data from the WHO Global Health Observatory, Ghana DHS Programme (ICF International), Ghana DHIMS2 (Ghana Health Service), and Ghana Statistical Service. No individual participant data were accessed. All inputs are de-identified district and regional summary statistics. Ethical review was not required for analysis of publicly available aggregate statistics; DHS data were accessed under the standard DHS Programme Data Use Agreement.

---

## 11. Citation

**APA:**
Ghanem, V. G. (2026). *Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 261-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and WASH Determinants.* GitHub. https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana

**BibTeX:**
```bibtex
@misc{ghanem2026malaria,
  author = {Ghanem, Valentine Golden},
  title  = {Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 261-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and WASH Determinants},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana}
}
```

A machine-readable citation is provided in `CITATION.cff`.

---

## 12. License

Code is released under the **MIT License** — see [LICENSE](LICENSE) for details.
Outputs and figures: **CC BY 4.0**.

---

## 13. Author & Contact

**Valentine Golden Ghanem**
Ghana COCOBOD Cocoa Clinic, Accra, Ghana
Email: valentineghanem@gmail.com
ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)

---

## 14. Acknowledgements

The author thanks the WHO for the Global Health Observatory malaria dataset, the DHS Programme and ICF International for Ghana DHS data, Ghana Health Service for DHIMS2 routine surveillance data, and the Ghana Statistical Service for Census district files and boundary geometries. Spatial analysis relied on esda, libpysal, spdep, and spatialreg. Ensemble modelling used XGBoost and scikit-learn; interpretability used SHAP. WorldPop provided open population surface data.
