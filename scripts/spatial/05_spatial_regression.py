#!/usr/bin/env python3
"""
05_spatial_regression.py — Ghana Malaria 260-District Analysis
SLM vs SEM selection via robust Lagrange Multiplier diagnostics.

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Outputs: data/processed/Spatial_Regression_Results.csv
Tenet 7 (Causal Clarity): Observational ecological study — no causal claims.
 Confounders listed: poverty, female education, healthcare access, rainfall.
 DAG: ITN → Transmission ← Parasitaemia ← WASH/Poverty/Education
"""

import os
import pickle
import pandas as pd
import numpy as np
from spreg import OLS, ML_Lag, ML_Error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(REPO_ROOT, "data", "processed")

CANDIDATE_FEATURES = [
 "itn_coverage", "parasitaemia_prev", "water_access_pct",
 "sanitation_pct", "u5mr", "poverty_index",
 "female_edu_secondary", "urban_pct",
]


def main() -> None:
 df = pd.read_csv(os.path.join(PROC, "Ghana_Malaria_260District_MasterDataset.csv"))
 inc_col = next(c for c in df.columns if "incidence" in c.lower())
 features = [c for c in CANDIDATE_FEATURES if c in df.columns]
 if len(features) < 2:
 num = df.select_dtypes(include="number").columns.tolist()
 features = [c for c in num if "incidence" not in c.lower()][:5]

 y = df[[inc_col]].values
 X = df[features].values
 print(f"[05] Outcome: {inc_col} | Predictors ({len(features)}): {features}")

 with open(os.path.join(PROC, "weights_rook.pkl"), "rb") as fh:
 w = pickle.load(fh)

 # OLS with spatial diagnostics (robust LM tests)
 ols = OLS(y, X, w=w, spat_diag=True,
 name_y=inc_col, name_x=features, name_ds="Ghana_Malaria_260")
 print(f"\n[05] OLS Baseline: R²={ols.r2:.4f} AIC={ols.aic:.2f}")
 print(f" LM-lag p={ols.lm_lag[1]:.4f} | Robust LM-lag p={ols.rlm_lag[1]:.4f}")
 print(f" LM-error p={ols.lm_error[1]:.4f} | Robust LM-error p={ols.rlm_error[1]:.4f}")

 # Model selection rule: compare robust LM p-values
 use_lag = ols.rlm_lag[1] < ols.rlm_error[1]
 ModelClass = ML_Lag if use_lag else ML_Error
 model_label = "Spatial Lag Model (ML)" if use_lag else "Spatial Error Model (ML)"
 print(f"\n[05] Selected: {model_label} (robust LM rule)")

 model = ModelClass(y, X, w=w, name_y=inc_col, name_x=features)
 print(f" Pseudo-R²={model.pr2:.4f} AIC={model.aic:.2f} LogLik={model.logll:.4f}")

 # /uq-flag — Tenet 6
 print(f"\n[/uq-flag] Spatial regression uncertainty:")
 print(f" Model selection based on robust LM — verify with Moran's I on residuals.")
 print(f" Ecological fallacy risk: district-level inference; not individual-level.")
 print(f" Confounders not controlled: rainfall, NDVI, elevation, service quality.")

 results = pd.DataFrame([{
 "selected_model": model_label,
 "pseudo_R2": round(model.pr2, 4),
 "AIC": round(model.aic, 4),
 "log_likelihood": round(model.logll, 4),
 "OLS_R2": round(ols.r2, 4),
 "OLS_AIC": round(ols.aic, 4),
 "n_districts": len(df),
 "predictors": "|".join(features),
 "weight_matrix": "Rook contiguity, row-standardised",
 }])
 path = os.path.join(PROC, "Spatial_Regression_Results.csv")
 results.to_csv(path, index=False)
 print(f"\n[05] ✓ Results → {path}")


if __name__ == "__main__":
 main()
