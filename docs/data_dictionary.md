# Data Dictionary — Ghana Malaria 260-District Master Dataset

**Project:** Spatial distribution, determinants, and ML-based risk prediction of malaria across 260 districts of Ghana 
**Principal Investigator:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra 
**Dataset Version:** v1.0 | April 2026 
**Geographic Unit:** Ghana New 260-District framework (administrative restructuring 2019–2020) 
**Reference Period:** 2018–2022 (primary) | DHS 2014 (spatially disaggregated covariates) 
**Encoding:** UTF-8 | Separator: comma | Row identifier: `district_id` 
**CRS:** WGS 84 (EPSG:4326) for all spatial outputs 
**Random Seed:** 42 (all stochastic operations — Tenet 8 Reproducibility) 

---

## Canonical Summary Statistics (QA-verified 2026-04-29)

| Statistic | Value | Source |
|-----------|-------|--------|
| N districts | 260 | GSS 2021 |
| Global Moran's I | 0.672 | 02_global_morans.py (KNN k=8, 999 perm) |
| Moran's z-score | 14.38 | p < 0.001 |
| LISA HH clusters | 36 | 03_bivariate_lisa.py (Rook, p<0.05) |
| Gi* Priority-1 districts | 38 | 04_getis_ord.py (|z|≥3.291) |
| XGBoost AUC-ROC | 0.923 | 07_xgboost_model.py (LODO-CV) |
| XGBoost Brier score | 0.093 | 07_xgboost_model.py |
| Top SHAP feature | itn_coverage (0.41) | 10_shap_interpretability.py |

---

## Column Definitions

### A. Geographic Identifiers

| Variable | Type | Unit | Definition | Source | Canonical / Notes |
|----------|------|------|------------|--------|-------------------|
| `district_id` | int | — | Unique district identifier (1–260); assigned by row order in Ghana_New_260_District.geojson | GSS 2021 | All 260 districts required |
| `district` | str | — | Official district name per Ghana New 260-District framework | GSS 2021 | Must match geojson `ADM2_EN` attribute exactly (case-sensitive) |
| `region` | str | — | Administrative region (16 regions post-2019 restructuring); used as grouping variable for LODO-CV | GSS 2021 | 16 unique values |
| `ecological_zone` | str | — | Ecological classification: `Guinea Savannah` · `Sudan Savannah` · `Transitional Forest-Savannah` · `Forest` · `Coastal/Urban` | Author classification (WorldPop + GSS boundaries) | 5 categories |
| `latitude` | float | decimal degrees | District centroid latitude (WGS 84) | Computed from geojson centroids | — |
| `longitude` | float | decimal degrees | District centroid longitude (WGS 84) | Computed from geojson centroids | — |

---

### B. Disease Outcome Variables

| Variable | Type | Unit | Definition | Source | Canonical / Notes |
|----------|------|------|------------|--------|-------------------|
| `incidence` | float | cases per 1,000 pop. at risk | Estimated malaria incidence (WHO GHO district-level interpolation, cross-validated with DHIMS2 OPD-confirmed cases) | WHO GHO 2022 / DHIMS2 2018–2022 | Mean: ~180/1,000; SD: ~90/1,000 |
| `parasitaemia_prev` | float | % | RDT-confirmed *Plasmodium falciparum* parasitaemia prevalence, children 6–59 months | Ghana DHS 2014 (spatially disaggregated to district level using WorldPop 2020 areal interpolation) | Canonical SHAP rank 2 (|SHAP|=0.38) |
| `mort_rate` | float | per 100,000 pop. | Estimated district-level malaria mortality rate | WHO GHO / DHIMS2 2018–2022 | — |
| `high_burden_binary` | int | 0/1 | Binary classification outcome: 1 = high-burden district (malaria incidence ≥ 100/1,000 at risk); 0 = lower burden | Author-derived (06_feature_engineering.py) | Threshold = 100/1,000; check class balance (expected ~25–40% positive) |

---

### C. Primary Exposure: Vector Control

| Variable | Type | Unit | Definition | Source | Canonical / Notes |
|----------|------|------|------------|--------|-------------------|
| `itn_coverage` | float | % | Proportion of households owning ≥1 insecticide-treated net (ITN); proxy for community-level vector control intensity | Ghana DHS 2014 (disaggregated to district level) | **Top SHAP predictor (|SHAP|=0.41)**; range [0, 100]; districts with ITN <40% carry steepest burden increase |

---

### D. WASH Variables

| Variable | Type | Unit | Definition | Source | Canonical / Notes |
|----------|------|------|------------|--------|-------------------|
| `water_access_pct` | float | % | Population with access to improved drinking water source (JMP definition: piped, borehole, protected spring/well, rainwater) | Ghana DHS 2014 / JMP 2020 | Canonical SHAP rank 3 (|SHAP|=0.27); districts <60% trigger WASH-malaria integrated programming recommendation |
| `sanitation_pct` | float | % | Population with access to improved sanitation facility (JMP definition) | Ghana DHS 2014 / JMP 2020 | Feature matrix col 4 |

---

### E. Socioeconomic and Demographic Variables

| Variable | Type | Unit | Definition | Source | Canonical / Notes |
|----------|------|------|------------|--------|-------------------|
| `u5mr` | float | per 1,000 live births | Under-five mortality rate; malaria-correlated child health indicator | Ghana DHS 2014 / Census 2021 | Feature matrix col 5 |
| `poverty_index` | float | 0–1 | Multidimensional Poverty Index (MPI); composite of education, health, and living standards deprivation | UNDP/Oxford Poverty and Human Development Initiative (OPHI) | Feature matrix col 6 |
| `female_edu_secondary` | float | % | Proportion of women aged 15–49 with at least secondary education completed | Ghana DHS 2014 | Feature matrix col 7; upstream determinant of health-seeking behaviour |
| `healthcare_density` | float | facilities per 100,000 pop. | Number of functional health facilities per 100,000 district population | Ghana Health Service (DHIMS2) / GHS facility registry | Feature matrix col 8 |
| `rainfall_mm` | float | mm/year | Mean annual rainfall (2010–2020 average) | CHIRPS v2.0 / WorldClim v2.1 | Feature matrix col 9; larval habitat driver |
| `urban_pct` | float | % | Proportion of district population residing in urban localities (GSS definition) | GSS Census 2021 | Feature matrix col 10 |
| `pop_density` | float | persons/km² | District population density | WorldPop 2020 / GSS Census 2021 | — |

---

### F. Spatial Analysis Outputs

| Variable | Type | Unit | Definition | Source | Canonical / Notes |
|----------|------|------|------------|--------|-------------------|
| `moran_local_I` | float | — | Local Moran's I statistic (univariate; incidence; KNN k=8, 999 permutations) | 02_global_morans.py (esda 2.6.0) | Positive = local clustering |
| `bv_lisa_quadrant` | str | HH/LL/HL/LH/NS | Bivariate LISA quadrant classification (ITN coverage × malaria incidence; Rook contiguity; 999 perm; α=0.05) | 03_bivariate_lisa.py | **Canonical HH=36**; valid labels: `High-High` · `Low-Low` · `High-Low` · `Low-High` · `Not Significant` |
| `bv_lisa_p` | float | p-value | Pseudo p-value from bivariate LISA permutation test | 03_bivariate_lisa.py | Flag: p < 0.05 = statistically significant cluster |
| `gi_z_score` | float | z-score | Getis-Ord Gi* z-score (KNN k=8, 999 permutations) | 04_getis_ord.py | Positive z = hotspot; negative z = coldspot |
| `gi_classification` | str | — | Gi* classification tier: `Hotspot_99.9pct` (Priority-1, \|z\|≥3.291) · `Hotspot_99pct` (\|z\|≥2.576) · `Hotspot_95pct` (\|z\|≥1.960) · `Coldspot_99pct` · `Coldspot_95pct` · `Not_Significant` | 04_getis_ord.py | **Canonical Priority-1=38**; /policy-bridge mandatory for Priority-1 districts |

---

### G. ML Model Outputs

| Variable | Type | Unit | Definition | Source | Canonical / Notes |
|----------|------|------|------------|--------|-------------------|
| `xgb_risk_score` | float | 0–1 | XGBoost predicted probability that district is high-burden (high_burden_binary=1); calibration validated via Brier score | 07_xgboost_model.py (XGBoost 2.0.3, SEED=42) | Canonical Brier=0.093; all values must be in [0, 1] |
| `xgb_risk_tier` | str | — | Risk classification: `Very High` (>0.80) · `High` (0.60–0.80) · `Moderate` (0.40–0.60) · `Low` (0.20–0.40) · `Very Low` (<0.20) | Author-derived from xgb_risk_score | — |
| `shap_itn_coverage` | float | SHAP value | Per-district SHAP value for itn_coverage; positive = ITN deficiency increases predicted burden | 10_shap_interpretability.py | Mean \|SHAP\|=0.41; SD threshold for instability: >0.05 |
| `shap_parasitaemia_prev` | float | SHAP value | Per-district SHAP value for parasitaemia_prev | 10_shap_interpretability.py | Mean \|SHAP\|=0.38 |
| `shap_water_access_pct` | float | SHAP value | Per-district SHAP value for water_access_pct | 10_shap_interpretability.py | Mean \|SHAP\|=0.27 |

---

### H. Data Provenance and Quality Flags

| Variable | Type | Unit | Definition | Source | Notes |
|----------|------|------|------------|--------|-------|
| `reporting_completeness_pct` | float | % | DHIMS2 OPD reporting completeness for the district-year used | DHIMS2 | Districts <80% flagged for sensitivity analysis |
| `data_source_incidence` | str | — | Data provenance for incidence variable | — | e.g., "WHO GHO 2022 (interpolated from regional estimates)" |
| `data_source_itn` | str | — | Data provenance for ITN coverage | — | e.g., "Ghana DHS 2014 disaggregated via WorldPop 2020" |
| `analysis_year` | int | year | Reference year of primary outcome data | — | 2022 |
| `dataset_version` | str | — | Dataset version string | — | "v1.0_2026-04-30" |

---

## Coding Conventions

- **Missing values:** Encoded as `NA` (R-style); do not mix with empty string or 0.
- **Percentages:** Stored as 0–100 (not 0–1 proportions), except `poverty_index` which is 0–1 by convention.
- **District name standardisation:** Must match `ADM2_EN` in `Ghana_New_260_District.geojson` exactly. Any mismatch causes GeoJSON join failure in dashboard.
- **Column naming:** `snake_case`; no spaces; units embedded in column name where unambiguous (e.g., `u5mr` = per 1,000 live births by convention).
- **DHIMS2 reporting completeness <80%:** Flagged in `reporting_completeness_pct`; sensitivity analysis comparing complete vs. all districts is mandatory before final manuscript submission.

## Spatial Disaggregation Note

Regional-level DHS 2014 estimates (16 administrative regions) were disaggregated to district level using areal interpolation weighted by WorldPop Ghana 2020 100m population density surface. Disaggregated values should be interpreted as modelled district-level estimates, not direct survey measurements. Uncertainty arising from disaggregation is acknowledged as a study limitation (STROBE checklist item 14b).

## Reproducibility Checklist

- `numpy.random.seed(42)` set in all scripts generating stochastic outputs.
- Spatial weights computed once and saved as `data/processed/weights_knn8.pkl` and `weights_rook.pkl`; do not re-generate without documenting.
- All model objects saved as pickle files in `data/models/`; re-run scripts regenerate identical output given identical input data and SEED=42.
- Full pipeline re-execution order: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10.
