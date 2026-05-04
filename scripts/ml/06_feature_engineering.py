#!/usr/bin/env python3
"""
06_feature_engineering.py — Ghana Malaria 260-District Analysis
Feature matrix construction and binary target creation for ML pipeline.

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Inputs : data/processed/Ghana_Malaria_260District_MasterDataset.csv
Outputs: data/processed/feature_matrix.csv, data/processed/target_vector.csv
Tenet 8 (Reproducibility): missing values imputed with column medians; logged.
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

# Ordered feature set — matches SHAP interpretability script
FEATURE_COLS = [
 "itn_coverage", # Insecticide-treated net coverage (%)
 "parasitaemia_prev", # Plasmodium falciparum parasitaemia (%)
 "water_access_pct", # Improved water source access (%)
 "sanitation_pct", # Improved sanitation (%)
 "u5mr", # Under-5 mortality rate (per 1,000 live births)
 "poverty_index", # Multidimensional poverty index
 "female_edu_secondary", # Female secondary education (%)
 "healthcare_density", # Health facilities per 100,000 population
 "rainfall_mm", # Mean annual rainfall (mm)
 "urban_pct", # Urban population (%)
]

HIGH_BURDEN_THRESHOLD = 100 # cases per 1,000 — top-quartile classification


def main() -> None:
 df = pd.read_csv(os.path.join(PROC, "Ghana_Malaria_260District_MasterDataset.csv"))
 assert len(df) == 260, f"Expected 260 rows, got {len(df)}"
 print(f"[06] Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

 inc_col = next(c for c in df.columns if "incidence" in c.lower())

 # Select available feature columns
 available = [c for c in FEATURE_COLS if c in df.columns]
 if len(available) < 3:
 # Fallback: all numeric columns except outcomes
 exclude = [c for c in df.columns if "incidence" in c.lower()
 or "mortality" in c.lower() or "burden" in c.lower()]
 available = [c for c in df.select_dtypes(include="number").columns
 if c not in exclude][:10]
 print(f"[06] Features used ({len(available)}): {available}")

 X = df[available].copy()

 # Binary outcome: high-burden district
 df["high_burden_binary"] = (df[inc_col] >= HIGH_BURDEN_THRESHOLD).astype(int)
 n_high = df["high_burden_binary"].sum()
 print(f"[06] High-burden districts (≥{HIGH_BURDEN_THRESHOLD}/1,000): "
 f"{n_high} ({100*n_high/len(df):.1f}%) — class balance check")

 # Missing value audit — Tenet 4 (Ground Truth Protocol)
 missing = X.isnull().sum()
 if missing.any():
 print("[06] Missing values detected — imputing with column medians:")
 for col, n in missing[missing > 0].items():
 pct = 100 * n / len(X)
 print(f" {col}: {n} missing ({pct:.1f}%)")
 X = X.fillna(X.median())

 # No infinite values
 inf_mask = X.isin([np.inf, -np.inf])
 if inf_mask.any().any():
 X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
 print("[06] ⚠ Infinite values replaced with column medians.")

 # Save outputs
 fm_path = os.path.join(PROC, "feature_matrix.csv")
 tv_path = os.path.join(PROC, "target_vector.csv")
 X.to_csv(fm_path, index=False)
 df[[inc_col, "high_burden_binary"]].to_csv(tv_path, index=False)
 print(f"\n[06] ✓ Feature matrix ({X.shape}) → {fm_path}")
 print(f"[06] ✓ Target vector → {tv_path}")


if __name__ == "__main__":
 main()
