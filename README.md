# Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 260-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and WASH Determinants

[![CI](https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**Reporting standard:** STROBE
**Date:** April 2026
**Status:** Manuscript in preparation

> Valentine Golden Ghanem (2026). *Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 260-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and WASH Determinants.* GitHub repository. https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana

---

## 1. Abstract

This study maps malaria burden across Ghana's 260 districts at unprecedented subnational resolution, integrating insecticide-treated net (ITN) coverage and WASH determinants. Spatial clustering identifies high-priority hotspot districts, and ensemble machine learning (XGBoost, Random Forest, CART, Logistic Regression) with Leave-One-District-Out (LODO) cross-validation produces calibrated district-level malaria risk predictions. SHAP TreeExplainer identifies ITN coverage as the dominant modifiable predictor.

---

## 2. Research Question & Aims

- **Primary:** Quantify the subnational distribution of malaria burden and identify priority hotspot districts.
- **Secondary:** (a) Detect ITN-deficit × malaria co-clusters using bivariate LISA; (b) build a LODO-CV ensemble ML pipeline for district-level risk prediction; (c) interpret model drivers using SHAP TreeExplainer; (d) tier hotspot districts by significance for programme prioritisation.

---

## 3. Methods Summary

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

## 4. Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| WHO Global Health Observatory | Malaria indicators | 2001–2023 | [who.int/data/gho](https://www.who.int/data/gho) (open) |
| Ghana DHS Programme | Subnational parasitaemia, ITN, WASH, U5MR | 2003–2022 | [dhsprogram.com](https://dhsprogram.com) (DUA required) |
| Ghana DHIMS2 | Routine malaria cases | 2018–2022 | Ghana Health Service |
| Ghana Statistical Service | District boundaries | 2021 | [statsghana.gov.gh](https://statsghana.gov.gh) |
| WorldPop | Population surface | 2020 | [worldpop.org](https://worldpop.org) (open) |

> DHS data accessed under signed Data Use Agreement (ICF International).

---

## 5. Key Findings

| Metric | Value |
|--------|-------|
| Global Moran's I (malaria incidence) | 0.672 (z=14.38, p<0.001) |
| Bivariate LISA High-High clusters (ITN deficit × malaria) | 36 |
| Priority-1 Gi* hotspot districts (99.9% CI) | 38 |
| All hotspot districts (all tiers) | 51 |
| XGBoost AUC-ROC | 0.923 (95% CI: 0.907–0.939), LODO-CV |
| Top SHAP predictor | ITN coverage (\|SHAP\|=0.41) |

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
│   └── figures/generate_figures.py # 300 DPI publication figures
├── app.py                          # Plotly Dash interactive application
├── dashboard/
│   └── Ghana_Malaria_260District_Dashboard.html
├── poster/
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
- Python 3.12 (see `requirements.txt` for pinned versions)
- R 4.3+ (for R scripts; see `renv.lock` or `analysis.R` header for pinned packages)
- Random seed: 42 throughout (set via `random_state=42` and `np.random.seed(42)`)
- Estimated runtime: ~15–20 minutes on a standard laptop (LODO-CV is N=260 folds)
- Tested on: Ubuntu 22.04 / macOS 14 / Windows 11 (CI: GitHub Actions)

### 7.2 Clone & install
```bash
git clone https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana.git
cd malaria-geospatial-ml-ghana
pip install -r requirements.txt
# For R scripts (optional):
Rscript -e "if (!requireNamespace('renv', quietly=TRUE)) install.packages('renv'); renv::restore()"
```

### 7.3 Run the analytical pipeline
```bash
# Spatial analysis (01–05)
python scripts/spatial/01_spatial_weights.py
python scripts/spatial/02_global_morans.py
python scripts/spatial/03_bivariate_lisa.py
python scripts/spatial/04_getis_ord.py
python scripts/spatial/05_spatial_regression.py

# ML pipeline (06–10)
python scripts/ml/06_feature_engineering.py
python scripts/ml/07_xgboost_model.py
python scripts/ml/08_random_forest.py
python scripts/ml/09_cart_logistic.py
python scripts/ml/10_shap_interpretability.py

# Figures
python scripts/figures/generate_figures.py
```

### 7.4 Run the test suite
```bash
pytest tests/ -v
```

### 7.5 Launch the interactive Dash application
```bash
python app.py
# Navigate to http://127.0.0.1:8050 in your browser
```

### 7.6 Open the static HTML dashboard
Open `dashboard/Ghana_Malaria_260District_Dashboard.html` in any modern browser. No server required.

---

## 8. Outputs

- **Interactive Dash app:** `app.py` — `python app.py` → http://127.0.0.1:8050
- **Static HTML dashboard:** `dashboard/Ghana_Malaria_260District_Dashboard.html`
- **Poster:** `poster/`
- **Master dataset:** `data/processed/master_district_data.csv`
- **Trained models + SHAP values:** `data/models/`
- **Figures:** `data/processed/figures/*.png` — 300 DPI

---

## 9. Reporting Standard

This study follows the **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology) reporting guideline for observational ecological studies.

---

## 10. Ethical Statement

This study used exclusively secondary data. No primary data collection from human participants was conducted. Ghana DHS data were accessed under a signed Data Use Agreement with ICF International. Ethical review was therefore not required for this analysis.

---

## 11. Citation

**APA:**
Ghanem, V. G. (2026). *Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 260-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and WASH Determinants*. GitHub. https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana

**BibTeX:**
```bibtex
@misc{ghanem2026malaria,
  author = {Ghanem, Valentine Golden},
  title  = {Geospatial Clustering and Machine Learning Prediction of Malaria Burden at 260-District Resolution in Ghana: Integrating Insecticide-Treated Net Coverage and WASH Determinants},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/malaria-geospatial-ml-ghana}
}
```

A machine-readable citation is provided in `CITATION.cff`.

---

## 12. License

Code is released under the **MIT License** — see [LICENSE](LICENSE) for details. Data outputs and figures: CC BY 4.0.

---

## 13. Author & Contact

- **Valentine Golden Ghanem**
  Ghana COCOBOD Cocoa Clinic, Accra, Ghana
  Email: [valentineghanem@gmail.com](mailto:valentineghanem@gmail.com)
  ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)

---

## 14. Acknowledgements

- **Ghana Demographic and Health Survey programme** (ICF International) for survey data access under signed Data Use Agreement.
- **Ghana Statistical Service** for the 2021 Population and Housing Census and administrative boundary data.
- **WHO Global Health Observatory** for national-level indicators.
- **WorldPop** for high-resolution population surfaces.
- **Ghana Health Service (DHIMS2)** for routine malaria surveillance data.
- **AIPOCH** (Anti-hallucination Pipeline for Open Computational Health) v6.0 quad-connector citation verification (PubMed · Consensus · Scholar · Scite).

---

*This README follows the AIPOCH v6.0 standardised research-output template (May 2026). All repository READMEs in the [valentineghanem-bit](https://github.com/valentineghanem-bit) organisation share this structure.*
