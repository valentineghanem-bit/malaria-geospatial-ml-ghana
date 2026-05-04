#!/usr/bin/env python3
"""
03_bivariate_lisa.py — Ghana Malaria 260-District Analysis
Bivariate Local Moran's I: ITN coverage (X) × malaria incidence (Y).
Spatial weight matrix: Rook contiguity | 999 permutations | alpha=0.05.

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Expected: HH=36, LL=22, HL=17, LH=9, NS=176
Outputs : data/processed/LISA_Results.csv
Tenet 5: Bivariate LISA follows Moran's I (run 02 first).
Tenet 6 (/uq-flag): p-values and cluster counts stated before interpretation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from esda.moran import Moran_Local_BV
from scipy.stats import zscore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(REPO_ROOT, "data", "processed")
SEED = 42
ALPHA = 0.05


def quadrant(z_x: float, z_lag: float, p: float) -> str:
 """Assign LISA quadrant label based on standardised values and significance."""
 if p >= ALPHA:
 return "Not Significant"
 if z_x >= 0 and z_lag >= 0:
 return "High-High" # ITN deficit co-located with high incidence
 if z_x < 0 and z_lag < 0:
 return "Low-Low" # Good ITN coverage co-located with low incidence
 if z_x >= 0 and z_lag < 0:
 return "High-Low" # Spatial outlier
 return "Low-High" # Spatial outlier


def main() -> None:
 df = pd.read_csv(os.path.join(PROC, "Ghana_Malaria_260District_MasterDataset.csv"))
 assert len(df) == 260

 inc_col = next(c for c in df.columns if "incidence" in c.lower())
 itn_col = next(c for c in df.columns if "itn" in c.lower() or "net" in c.lower())
 dist_col = next(c for c in df.columns if "district" in c.lower() or "name" in c.lower())
 print(f"[03] X (exposure): {itn_col} | Y (outcome): {inc_col}")

 # Standardise to zero mean, unit variance before bivariate LISA
 x = zscore(df[itn_col].values)
 y = zscore(df[inc_col].values)

 with open(os.path.join(PROC, "weights_rook.pkl"), "rb") as fh:
 w = pickle.load(fh)

 np.random.seed(SEED)
 lbv = Moran_Local_BV(x, y, w, permutations=999, seed=SEED)

 res = pd.DataFrame({
 "district": df[dist_col].values,
 "local_I": lbv.Is,
 "p_sim": lbv.p_sim,
 "z_x_standardised": x,
 "spatial_lag": lbv.EI_sim,
 "itn_raw_pct": df[itn_col].values,
 "incidence_raw": df[inc_col].values,
 })
 res["LISA_Cluster"] = [
 quadrant(row.z_x_standardised, row.spatial_lag, row.p_sim)
 for row in res.itertuples()
 ]

 counts = res["LISA_Cluster"].value_counts()
 print("\n[/uq-flag] Bivariate LISA Summary (p<0.05, Rook contiguity, 999 permutations):")
 for q in ["High-High", "Low-Low", "High-Low", "Low-High", "Not Significant"]:
 n = counts.get(q, 0)
 print(f" {q:22s}: {n:3d} ({100*n/len(res):.1f}%)")

 out = os.path.join(PROC, "LISA_Results.csv")
 res.to_csv(out, index=False)
 print(f"\n[03] ✓ LISA results → {out}")

 hh = counts.get("High-High", 0)
 status = "✓" if hh == 36 else "⚠ check ITN column and Rook weights"
 print(f"[03] HH clusters: {hh} (expected 36) {status}")


if __name__ == "__main__":
 main()
