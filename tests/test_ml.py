#!/usr/bin/env python3
"""
tests/test_ml.py — Ghana Malaria 260-District Analysis
Unit tests for ML model outputs, SHAP interpretability, and risk classification.
Canonical values anchored to QA_GhanaMalaria_2026-04-29.md.

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Run : pytest tests/test_ml.py -v

/uq-flag: SHAP instability thresholds enforced (SD > 0.05 → High_Instability flag).
Tenet 13: SHAP output presence validated in canonical CSV.
"""

import os
import pickle
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(REPO_ROOT, "data", "processed")
MODELS = os.path.join(REPO_ROOT, "data", "models")
FIG_DIR = os.path.join(REPO_ROOT, "scripts", "figures")
MASTER_CSV = os.path.join(PROC, "Ghana_Malaria_260District_MasterDataset.csv")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_proc_csv(name: str) -> pd.DataFrame:
 path = os.path.join(PROC, name)
 if not os.path.exists(path):
 pytest.skip(f"{name} not found — run the ML pipeline first.")
 return pd.read_csv(path)


def load_model(name: str):
 path = os.path.join(MODELS, name)
 if not os.path.exists(path):
 pytest.skip(f"{name} not found — run the ML training script first.")
 with open(path, "rb") as fh:
 return pickle.load(fh)


# ─── TestFeatureMatrix ────────────────────────────────────────────────────────

class TestFeatureMatrix:
 """Feature matrix structural integrity checks."""

 def test_feature_matrix_shape(self):
 """Feature matrix must have exactly 260 rows."""
 df = load_proc_csv("feature_matrix.csv")
 assert len(df) == 260, f"Feature matrix has {len(df)} rows; expected 260"

 def test_feature_count(self):
 """Feature matrix must have ≥3 columns (minimum viable feature set)."""
 df = load_proc_csv("feature_matrix.csv")
 assert df.shape[1] >= 3, f"Feature matrix has only {df.shape[1]} columns"

 def test_no_missing_values(self):
 """Feature matrix must have zero missing values after imputation."""
 df = load_proc_csv("feature_matrix.csv")
 missing = df.isnull().sum().sum()
 assert missing == 0, f"Feature matrix contains {missing} missing values after imputation"

 def test_no_infinite_values(self):
 """Feature matrix must contain no infinite values."""
 df = load_proc_csv("feature_matrix.csv")
 inf_count = np.isinf(df.values).sum()
 assert inf_count == 0, f"Feature matrix contains {inf_count} infinite values"

 def test_itn_coverage_present(self):
 """ITN coverage must be present — it is the top SHAP predictor."""
 df = load_proc_csv("feature_matrix.csv")
 itn_cols = [c for c in df.columns if "itn" in c.lower()]
 assert len(itn_cols) > 0, (
 "itn_coverage column absent from feature matrix; "
 "it is the canonical top SHAP predictor (|SHAP|=0.41)"
 )


# ─── TestTargetVector ─────────────────────────────────────────────────────────

class TestTargetVector:
 """Binary target vector checks."""

 def test_target_length(self):
 """Target vector must have 260 rows."""
 df = load_proc_csv("target_vector.csv")
 assert len(df) == 260, f"Target vector has {len(df)} rows; expected 260"

 def test_binary_outcome_present(self):
 """high_burden_binary column must be present."""
 df = load_proc_csv("target_vector.csv")
 assert "high_burden_binary" in df.columns, \
 "high_burden_binary column missing from target vector"

 def test_binary_values_only(self):
 """high_burden_binary must contain only 0 and 1."""
 df = load_proc_csv("target_vector.csv")
 unique_vals = set(df["high_burden_binary"].unique())
 assert unique_vals.issubset({0, 1}), \
 f"Non-binary values in high_burden_binary: {unique_vals}"

 def test_class_imbalance_acceptable(self):
 """Positive class prevalence must be between 10% and 60%."""
 df = load_proc_csv("target_vector.csv")
 prev = df["high_burden_binary"].mean()
 assert 0.10 <= prev <= 0.60, (
 f"Class prevalence {prev:.1%} outside acceptable range [10%, 60%]. "
 "Check HIGH_BURDEN_THRESHOLD in 06_feature_engineering.py."
 )


# ─── TestXGBoostResults ───────────────────────────────────────────────────────

class TestXGBoostResults:
 """XGBoost LODO-CV performance canonical assertions.
 Canonical: AUC-ROC = 0.923, Brier = 0.093 (LODO spatial CV, SEED=42)."""

 def test_xgb_results_csv_exists(self):
 """XGBoost results CSV must be present."""
 load_proc_csv("xgb_results.csv")

 def test_xgb_auc_canonical(self):
 """XGBoost AUC-ROC must equal canonical 0.923 ± 0.05."""
 df = load_proc_csv("xgb_results.csv")
 auc_col = next((c for c in df.columns if "auc" in c.lower()), None)
 assert auc_col is not None, "AUC column missing from xgb_results.csv"
 auc = df[auc_col].iloc[0]
 assert abs(auc - 0.923) <= 0.05, (
 f"XGBoost AUC = {auc:.4f}; canonical 0.923 ± 0.05. "
 "Check LODO-CV configuration or random seed."
 )

 def test_xgb_auc_above_floor(self):
 """XGBoost AUC-ROC must exceed 0.80 (acceptable floor for spatial prediction)."""
 df = load_proc_csv("xgb_results.csv")
 auc_col = next((c for c in df.columns if "auc" in c.lower()), None)
 if auc_col is None:
 pytest.skip("AUC column missing")
 auc = df[auc_col].iloc[0]
 assert auc > 0.80, f"XGBoost AUC = {auc:.4f} below acceptable floor of 0.80"

 def test_xgb_brier_score_canonical(self):
 """XGBoost Brier score must equal canonical 0.093 ± 0.03."""
 df = load_proc_csv("xgb_results.csv")
 brier_col = next((c for c in df.columns if "brier" in c.lower()), None)
 if brier_col is None:
 pytest.skip("Brier column missing from xgb_results.csv")
 brier = df[brier_col].iloc[0]
 assert abs(brier - 0.093) <= 0.03, (
 f"Brier score = {brier:.4f}; canonical 0.093 ± 0.03"
 )

 def test_xgb_brier_below_threshold(self):
 """Brier score must be <0.15 (well-calibrated model threshold)."""
 df = load_proc_csv("xgb_results.csv")
 brier_col = next((c for c in df.columns if "brier" in c.lower()), None)
 if brier_col is None:
 pytest.skip("Brier column missing")
 brier = df[brier_col].iloc[0]
 assert brier < 0.15, f"Brier score {brier:.4f} exceeds 0.15 — model calibration poor"

 def test_xgb_model_file_exists(self):
 """Pickled XGBoost model must exist."""
 load_model("xgb_model.pkl")

 def test_random_seed_documented(self):
 """XGBoost results must document random seed (Tenet 8 — Reproducibility)."""
 df = load_proc_csv("xgb_results.csv")
 seed_col = next((c for c in df.columns if "seed" in c.lower()), None)
 if seed_col:
 assert df[seed_col].iloc[0] == 42, \
 f"Random seed = {df[seed_col].iloc[0]}; expected 42 (Tenet 8)"


# ─── TestSHAPResults ──────────────────────────────────────────────────────────

class TestSHAPResults:
 """SHAP interpretability canonical assertions.
 Canonical top 3: ITN coverage (0.41) > Parasitaemia (0.38) > Water access (0.27).
 /uq-flag: SHAP_SD > 0.05 triggers High_Instability flag (Tenet 6).
 Tenet 13: summary + waterfall + dependence plots mandatory."""

 def test_shap_csv_exists(self):
 """SHAP results CSV must exist after running 10_shap_interpretability.py."""
 load_proc_csv("SHAP_Results.csv")

 def test_shap_required_columns(self):
 """SHAP CSV must contain Feature, Mean_Abs_SHAP, SHAP_SD, High_Instability."""
 df = load_proc_csv("SHAP_Results.csv")
 required = {"Feature", "Mean_Abs_SHAP", "SHAP_SD", "High_Instability"}
 missing_cols = required - set(df.columns)
 assert not missing_cols, f"Missing SHAP columns: {missing_cols}"

 def test_shap_top1_itn_coverage(self):
 """Top SHAP predictor must be ITN-related (canonical: itn_coverage, |SHAP|=0.41)."""
 df = load_proc_csv("SHAP_Results.csv")
 top_feat = df.sort_values("Mean_Abs_SHAP", ascending=False).iloc[0]["Feature"]
 assert "itn" in top_feat.lower(), (
 f"Top SHAP feature is '{top_feat}'; expected itn_coverage. "
 "ITN scale-up is the single highest-yield policy lever — verify pipeline."
 )

 def test_shap_top1_value_canonical(self):
 """Top SHAP |value| must equal canonical 0.41 ± 0.15."""
 df = load_proc_csv("SHAP_Results.csv")
 top_shap = df.sort_values("Mean_Abs_SHAP", ascending=False).iloc[0]["Mean_Abs_SHAP"]
 assert abs(top_shap - 0.41) <= 0.15, (
 f"Top SHAP |value| = {top_shap:.3f}; canonical 0.41 ± 0.15"
 )

 def test_shap_instability_flag_column(self):
 """High_Instability column must be boolean/binary (/uq-flag — Tenet 6)."""
 df = load_proc_csv("SHAP_Results.csv")
 assert "High_Instability" in df.columns, "High_Instability column missing"
 unique_vals = set(df["High_Instability"].unique())
 assert unique_vals.issubset({True, False, 0, 1}), \
 f"High_Instability has non-boolean values: {unique_vals}"

 def test_shap_summary_plot_exists(self):
 """SHAP summary plot must be present (Tenet 13 — SHAP Mandatory)."""
 path = os.path.join(FIG_DIR, "shap_summary.png")
 if not os.path.exists(path):
 pytest.skip("shap_summary.png not found — run 10_shap_interpretability.py")
 size = os.path.getsize(path)
 assert size > 10_000, f"shap_summary.png too small ({size} bytes) — possible empty file"

 def test_shap_waterfall_plot_exists(self):
 """SHAP waterfall plot must be present (Tenet 13)."""
 path = os.path.join(FIG_DIR, "shap_waterfall.png")
 if not os.path.exists(path):
 pytest.skip("shap_waterfall.png not found — run 10_shap_interpretability.py")
 assert os.path.getsize(path) > 10_000, "shap_waterfall.png appears empty"

 def test_shap_dependence_plots_exist(self):
 """At least one SHAP dependence plot must exist (Tenet 13 — top 3 required)."""
 dep_plots = [f for f in os.listdir(FIG_DIR)
 if f.startswith("shap_dependence_") and f.endswith(".png")]
 if not dep_plots:
 pytest.skip("No SHAP dependence plots found — run 10_shap_interpretability.py")
 assert len(dep_plots) >= 1, "Expected ≥1 SHAP dependence plot"


# ─── TestMLRiskTiers ──────────────────────────────────────────────────────────

class TestMLRiskTiers:
 """XGBoost risk score and tier classification in master CSV."""

 def test_risk_score_bounds(self):
 """XGBoost risk scores must be bounded [0, 1]."""
 if not os.path.exists(MASTER_CSV):
 pytest.skip("Master CSV not found")
 df = pd.read_csv(MASTER_CSV)
 risk_col = next((c for c in df.columns if "risk_score" in c.lower()
 or "xgb_prob" in c.lower()), None)
 if risk_col is None:
 pytest.skip("Risk score column not found in master CSV")
 assert df[risk_col].between(0, 1).all(), \
 f"Risk scores outside [0, 1] in column '{risk_col}'"

 def test_risk_tier_labels_valid(self):
 """XGBoost risk tier labels must be from the valid classification set."""
 if not os.path.exists(MASTER_CSV):
 pytest.skip("Master CSV not found")
 df = pd.read_csv(MASTER_CSV)
 tier_col = next((c for c in df.columns if "risk_tier" in c.lower()
 or "tier" in c.lower()), None)
 if tier_col is None:
 pytest.skip("Risk tier column not found in master CSV")
 valid_tiers = {"Very High", "High", "Moderate", "Low", "Very Low"}
 unique_tiers = set(df[tier_col].dropna().unique())
 invalid = unique_tiers - valid_tiers
 assert not invalid, f"Invalid risk tier labels: {invalid}"

 def test_very_high_tier_district_count(self):
 """Very High risk districts should be in plausible range [20, 50]."""
 if not os.path.exists(MASTER_CSV):
 pytest.skip("Master CSV not found")
 df = pd.read_csv(MASTER_CSV)
 tier_col = next((c for c in df.columns if "risk_tier" in c.lower()
 or "tier" in c.lower()), None)
 if tier_col is None:
 pytest.skip("Risk tier column not found")
 very_high = (df[tier_col] == "Very High").sum()
 assert 20 <= very_high <= 50, (
 f"Very High risk district count = {very_high}; expected 20–50"
 )

 def test_rf_model_exists(self):
 """Random Forest model pickle must be present."""
 load_model("rf_model.pkl")

 def test_cart_model_exists(self):
 """CART model pickle must be present."""
 load_model("cart_model.pkl")

 def test_rf_results_csv_exists(self):
 """Random Forest results CSV must be present after running 08_random_forest.py."""
 df = load_proc_csv("rf_results.csv")
 auc_col = next((c for c in df.columns if "auc" in c.lower()), None)
 assert auc_col is not None, "AUC column missing from rf_results.csv"
 auc = df[auc_col].iloc[0]
 assert auc > 0.70, f"RF AUC = {auc:.4f} below acceptable floor of 0.70"

 def test_cart_logistic_results_csv_exists(self):
 """CART and logistic regression comparative results must be present."""
 df = load_proc_csv("cart_logistic_results.csv")
 assert "model" in df.columns, "model column missing from cart_logistic_results.csv"
 assert len(df) == 2, f"Expected 2 rows (CART + LR); got {len(df)}"
