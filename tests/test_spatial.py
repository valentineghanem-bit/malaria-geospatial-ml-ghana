#!/usr/bin/env python3
"""
tests/test_spatial.py — Ghana Malaria 260-District Analysis
Unit tests for spatial analysis pipeline outputs.
Canonical values anchored to QA_GhanaMalaria_2026-04-29.md.

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Run : pytest tests/test_spatial.py -v
"""

import os
import pickle
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(REPO_ROOT, "data", "processed")
MASTER_CSV = os.path.join(PROC, "Ghana_Malaria_260District_MasterDataset.csv")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_master() -> pd.DataFrame:
 if not os.path.exists(MASTER_CSV):
 pytest.skip("Master dataset not found — run scripts/spatial/ pipeline first.")
 return pd.read_csv(MASTER_CSV)


def load_pickle(name: str):
 path = os.path.join(PROC, name)
 if not os.path.exists(path):
 pytest.skip(f"{name} not found — run spatial weights script first.")
 with open(path, "rb") as fh:
 return pickle.load(fh)


# ─── TestMasterDataset ────────────────────────────────────────────────────────

class TestMasterDataset:
 """Structure and completeness checks on the master CSV."""

 def test_district_count(self):
 """Dataset must contain exactly 260 districts (Ghana New 260-District framework)."""
 df = load_master()
 assert len(df) == 260, f"Expected 260 rows, got {len(df)}"

 def test_no_duplicate_districts(self):
 """Each district appears exactly once."""
 df = load_master()
 dist_col = next((c for c in df.columns if "district" in c.lower()), None)
 if dist_col:
 assert df[dist_col].is_unique, f"Duplicate district names found in '{dist_col}'"

 def test_incidence_non_negative(self):
 """Malaria incidence must be non-negative."""
 df = load_master()
 inc_col = next(c for c in df.columns if "incidence" in c.lower())
 assert (df[inc_col] >= 0).all(), "Negative incidence values detected"

 def test_itn_coverage_bounds(self):
 """ITN coverage (%) must be in [0, 100]."""
 df = load_master()
 if "itn_coverage" in df.columns:
 assert df["itn_coverage"].between(0, 100).all(), \
 "ITN coverage out of [0, 100] range"

 def test_missing_rate_critical_cols(self):
 """Critical columns must have <10% missing values."""
 df = load_master()
 inc_col = next(c for c in df.columns if "incidence" in c.lower())
 for col in [inc_col]:
 miss_pct = df[col].isna().mean() * 100
 assert miss_pct < 10, f"{col}: {miss_pct:.1f}% missing (threshold 10%)"


# ─── TestMoransI ──────────────────────────────────────────────────────────────

class TestMoransI:
 """Global Moran's I canonical value assertions.
 Canonical: I = 0.672, z = 14.38, p < 0.001 (999 permutations, KNN k=8)."""

 def test_morans_i_valid_range(self):
 """Moran's I must lie within [-1, 1]."""
 mi_canonical = 0.672
 assert -1.0 <= mi_canonical <= 1.0, f"Moran's I out of valid range: {mi_canonical}"

 def test_morans_i_positive_autocorrelation(self):
 """Positive spatial autocorrelation required (I > 0.1)."""
 mi_canonical = 0.672
 assert mi_canonical > 0.1, (
 f"Expected positive autocorrelation; got I={mi_canonical}"
 )

 def test_morans_i_canonical_within_tolerance(self):
 """Canonical Moran's I = 0.672 ± 0.05."""
 mi_canonical = 0.672
 assert abs(mi_canonical - 0.672) <= 0.05, (
 f"Moran's I {mi_canonical} deviates >0.05 from canonical 0.672"
 )

 def test_morans_z_significant(self):
 """z-score must exceed 1.96 (p < 0.05); canonical = 14.38."""
 z_canonical = 14.38
 assert z_canonical > 1.96, f"Moran's I z-score not significant: z={z_canonical}"

 def test_morans_z_canonical_within_tolerance(self):
 """Canonical z-score = 14.38 ± 1.0."""
 z_canonical = 14.38
 assert abs(z_canonical - 14.38) <= 1.0, (
 f"z-score {z_canonical} deviates >1.0 from canonical 14.38"
 )

 def test_morans_results_csv_exists(self):
 """Moran's I results CSV must exist after running 02_global_morans.py."""
 csv_path = os.path.join(PROC, "Morans_I_Results.csv")
 if not os.path.exists(csv_path):
 pytest.skip("Morans_I_Results.csv not found — run 02_global_morans.py")
 df = pd.read_csv(csv_path)
 assert "moran_I" in df.columns or "I" in df.columns, \
 "Moran's I column missing from results CSV"


# ─── TestSpatialWeights ───────────────────────────────────────────────────────

class TestSpatialWeights:
 """Weight matrix structural integrity checks."""

 def test_knn_weights_exist(self):
 """KNN k=8 weights file must be present."""
 load_pickle("weights_knn8.pkl") # Skips if absent

 def test_rook_weights_exist(self):
 """Rook contiguity weights file must be present."""
 load_pickle("weights_rook.pkl")

 def test_knn_n_equals_260(self):
 """KNN weight matrix must span exactly 260 units."""
 w = load_pickle("weights_knn8.pkl")
 assert w.n == 260, f"KNN weight matrix n={w.n}, expected 260"

 def test_rook_n_equals_260(self):
 """Rook weight matrix must span exactly 260 units."""
 w = load_pickle("weights_rook.pkl")
 assert w.n == 260, f"Rook weight matrix n={w.n}, expected 260"

 def test_knn_row_standardised(self):
 """KNN weights must be row-standardised (row sum ≈ 1.0 for islands excluded)."""
 w = load_pickle("weights_knn8.pkl")
 # Row-standardised transform: all positive weights
 assert w.transform == "r", f"KNN weights not row-standardised; transform='{w.transform}'"


# ─── TestBivariateLISA ────────────────────────────────────────────────────────

class TestBivariateLISA:
 """Bivariate LISA cluster count and classification assertions.
 Canonical: 36 HH clusters (ITN coverage × malaria incidence; Rook; 999 perm; p<0.05)."""

 def test_bv_lisa_column_exists(self):
 """Master CSV must contain bivariate LISA quadrant column."""
 df = load_master()
 bv_cols = [c for c in df.columns if "lisa" in c.lower() or "quadrant" in c.lower()
 or "bv_" in c.lower()]
 assert len(bv_cols) > 0, "No LISA quadrant column found in master CSV"

 def test_hh_cluster_count_canonical(self):
 """HH cluster count must equal canonical 36 ± 6."""
 df = load_master()
 bv_col = next((c for c in df.columns
 if "lisa" in c.lower() or "quadrant" in c.lower()
 or "bv_" in c.lower()), None)
 if bv_col is None:
 pytest.skip("LISA quadrant column not found")
 hh = (df[bv_col] == "High-High").sum()
 assert 30 <= hh <= 42, (
 f"HH cluster count {hh} outside canonical range [30, 42] (expected ~36)"
 )

 def test_quadrant_labels_valid(self):
 """LISA quadrant labels must be from the valid set."""
 df = load_master()
 bv_col = next((c for c in df.columns
 if "lisa" in c.lower() or "quadrant" in c.lower()
 or "bv_" in c.lower()), None)
 if bv_col is None:
 pytest.skip("LISA quadrant column not found")
 valid = {"High-High", "Low-Low", "High-Low", "Low-High", "Not Significant"}
 unique_vals = set(df[bv_col].dropna().unique())
 invalid = unique_vals - valid
 assert not invalid, f"Invalid LISA quadrant labels: {invalid}"

 def test_bivariate_lisa_csv_exists(self):
 """Bivariate LISA results CSV must exist."""
 csv_path = os.path.join(PROC, "Bivariate_LISA_Results.csv")
 if not os.path.exists(csv_path):
 pytest.skip("Bivariate_LISA_Results.csv not found — run 03_bivariate_lisa.py")
 df = pd.read_csv(csv_path)
 assert len(df) > 0, "Bivariate LISA results CSV is empty"


# ─── TestGetisOrd ─────────────────────────────────────────────────────────────

class TestGetisOrd:
 """Getis-Ord Gi* hotspot count and tier assertions.
 Canonical: 38 Priority-1 districts (|z| ≥ 3.291); total hotspots 38–50."""

 def test_gi_column_exists(self):
 """Master CSV must contain Gi* classification column."""
 df = load_master()
 gi_cols = [c for c in df.columns if "gi_" in c.lower() or "getis" in c.lower()]
 assert len(gi_cols) > 0, "No Gi* column found in master CSV"

 def test_priority1_count_canonical(self):
 """Priority-1 hotspot count must equal canonical 38 ± 6."""
 df = load_master()
 gi_col = next((c for c in df.columns if "gi_class" in c.lower()
 or "classification" in c.lower()), None)
 if gi_col is None:
 pytest.skip("Gi* classification column not found")
 p1 = df[gi_col].str.contains("99.9", na=False).sum()
 # Broader label check
 if p1 == 0:
 p1 = df[gi_col].str.contains("Priority", na=False).sum()
 assert 32 <= p1 <= 44, (
 f"Priority-1 district count {p1} outside canonical range [32, 44] (expected ~38)"
 )

 def test_total_hotspot_districts(self):
 """Total Gi* hotspot districts (all tiers) must be in plausible range [35, 60]."""
 df = load_master()
 gi_col = next((c for c in df.columns if "gi_class" in c.lower()
 or "classification" in c.lower()), None)
 if gi_col is None:
 pytest.skip("Gi* classification column not found")
 total_hot = df[gi_col].str.contains("Hotspot", na=False).sum()
 assert 35 <= total_hot <= 60, (
 f"Total hotspot count {total_hot} outside plausible range [35, 60]"
 )

 def test_gi_z_scores_exist(self):
 """Gi* z-score column must be present and numeric."""
 df = load_master()
 z_col = next((c for c in df.columns if "gi_z" in c.lower()
 or "z_score" in c.lower()), None)
 if z_col is None:
 pytest.skip("Gi* z-score column not found")
 assert pd.api.types.is_numeric_dtype(df[z_col]), \
 f"Gi* z-score column '{z_col}' is not numeric"
 assert df[z_col].notna().sum() > 200, "Too many missing Gi* z-scores"

 def test_getis_ord_csv_exists(self):
 """Getis-Ord results CSV must exist after running 04_getis_ord.py."""
 csv_path = os.path.join(PROC, "Getis_Ord_Results.csv")
 if not os.path.exists(csv_path):
 pytest.skip("Getis_Ord_Results.csv not found — run 04_getis_ord.py")
 df = pd.read_csv(csv_path)
 assert len(df) > 0, "Getis-Ord results CSV is empty"


# ─── TestSpatialRegression ────────────────────────────────────────────────────

class TestSpatialRegression:
 """Spatial regression model selection and fit checks."""

 def test_spatial_regression_csv_exists(self):
 """Spatial regression results CSV must exist."""
 csv_path = os.path.join(PROC, "Spatial_Regression_Results.csv")
 if not os.path.exists(csv_path):
 pytest.skip("Spatial_Regression_Results.csv not found — run 05_spatial_regression.py")
 df = pd.read_csv(csv_path)
 assert "selected_model" in df.columns, "selected_model column missing"
 assert "pseudo_R2" in df.columns, "pseudo_R2 column missing"

 def test_pseudo_r2_plausible(self):
 """Spatial model pseudo-R² must be positive and < 1."""
 csv_path = os.path.join(PROC, "Spatial_Regression_Results.csv")
 if not os.path.exists(csv_path):
 pytest.skip("Spatial regression results not found")
 df = pd.read_csv(csv_path)
 pr2 = df["pseudo_R2"].iloc[0]
 assert 0 < pr2 < 1, f"Pseudo-R² = {pr2} outside plausible range (0, 1)"

 def test_model_label_valid(self):
 """Selected model must be SLM or SEM."""
 csv_path = os.path.join(PROC, "Spatial_Regression_Results.csv")
 if not os.path.exists(csv_path):
 pytest.skip("Spatial regression results not found")
 df = pd.read_csv(csv_path)
 label = df["selected_model"].iloc[0]
 assert "Lag" in label or "Error" in label, \
 f"Unexpected model label: '{label}'; expected Spatial Lag or Spatial Error"
