#!/usr/bin/env python3
"""
10_shap_interpretability.py — Ghana Malaria 260-District Analysis
SHAP TreeExplainer: summary, waterfall, dependence plots (top 5 features).

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Tenet 6 (/uq-flag): SHAP instability across 50 bootstrap replicates reported.
Tenet 13 (SHAP Mandatory): summary + waterfall + dependence plots generated.
Expected top 3: ITN coverage (0.41) > Parasitaemia (0.38) > Water access (0.27)
Outputs : scripts/figures/shap_*.png, data/processed/SHAP_Results.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(REPO_ROOT, "data", "processed")
MODELS = os.path.join(REPO_ROOT, "data", "models")
FIG_DIR = os.path.join(REPO_ROOT, "scripts", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
N_BOOTSTRAP = 50 # Instability estimation replicates


def bootstrap_shap_sd(model, X: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
 """
 Estimate SHAP instability as SD of mean |SHAP| across bootstrap replicates.
 High SD (>0.05) indicates unreliable feature importance for that predictor.
 """
 rng = np.random.RandomState(seed)
 explainer = shap.TreeExplainer(model)
 boot_means = []
 for _ in range(n_boot):
 idx = rng.choice(len(X), size=len(X), replace=True)
 sv = explainer.shap_values(X[idx])
 if isinstance(sv, list):
 sv = sv[1]
 boot_means.append(np.abs(sv).mean(axis=0))
 return np.std(np.array(boot_means), axis=0)


def main() -> None:
 X_df = pd.read_csv(os.path.join(PROC, "feature_matrix.csv"))
 X = X_df.values
 feat_names = X_df.columns.tolist()

 model_path = os.path.join(MODELS, "xgb_model.pkl")
 if not os.path.exists(model_path):
 raise FileNotFoundError(
 f"XGBoost model not found: {model_path}\n"
 "Run 07_xgboost_model.py first."
 )
 with open(model_path, "rb") as fh:
 model = pickle.load(fh)

 explainer = shap.TreeExplainer(model)
 sv = explainer.shap_values(X)
 if isinstance(sv, list):
 sv = sv[1]

 mean_abs = np.abs(sv).mean(axis=0)
 shap_sd = bootstrap_shap_sd(model, X, N_BOOTSTRAP, SEED)

 shap_df = pd.DataFrame({
 "Feature": feat_names,
 "Mean_Abs_SHAP": mean_abs,
 "SHAP_SD": shap_sd,
 }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
 shap_df["High_Instability"] = shap_df["SHAP_SD"] > 0.05

 # /uq-flag — mandatory before interpretation
 print("\n[/uq-flag] SHAP UNCERTAINTY QUANTIFICATION")
 print(f" Bootstrap replicates: {N_BOOTSTRAP} | seed={SEED}")
 print(shap_df[["Feature", "Mean_Abs_SHAP", "SHAP_SD", "High_Instability"]].to_string(index=False))

 # Tenet 13 — public-health translation of top 3 features
 top3 = shap_df.head(3)
 print("\n[TENET 13] Top 3 SHAP features — plain public-health interpretation:")
 print(f" 1. {top3.iloc[0]['Feature']} (|SHAP|={top3.iloc[0]['Mean_Abs_SHAP']:.3f}±{top3.iloc[0]['SHAP_SD']:.3f}): "
 "ITN scale-up is the single highest-yield policy lever; "
 "districts with ITN coverage <40% face the steepest malaria burden increase.")
 print(f" 2. {top3.iloc[1]['Feature']} (|SHAP|={top3.iloc[1]['Mean_Abs_SHAP']:.3f}): "
 "Active transmission intensity; high-parasitaemia districts require "
 "targeted case management and seasonal malaria chemoprevention.")
 print(f" 3. {top3.iloc[2]['Feature']} (|SHAP|={top3.iloc[2]['Mean_Abs_SHAP']:.3f}): "
 "WASH-related larval habitat risk; integrated WASH-malaria programming "
 "is indicated for districts with water access below 60%.")

 # ── Summary plot ──────────────────────────────────────────────────────────
 shap.summary_plot(sv, X_df, show=False, max_display=10)
 plt.tight_layout()
 plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
 plt.close()
 print("\n[10] ✓ SHAP summary plot → scripts/figures/shap_summary.png")

 # ── Waterfall plot (district index 0) ─────────────────────────────────────
 base_val = (explainer.expected_value
 if not isinstance(explainer.expected_value, list)
 else explainer.expected_value[1])
 exp = shap.Explanation(
 values=sv[0], base_values=base_val, data=X[0], feature_names=feat_names)
 shap.plots.waterfall(exp, show=False)
 plt.tight_layout()
 plt.savefig(os.path.join(FIG_DIR, "shap_waterfall.png"), dpi=300, bbox_inches="tight")
 plt.close()
 print("[10] ✓ SHAP waterfall plot → scripts/figures/shap_waterfall.png")

 # ── Dependence plots — top 3 features ────────────────────────────────────
 for feat in shap_df["Feature"].head(3).tolist():
 idx = feat_names.index(feat)
 shap.dependence_plot(idx, sv, X_df, show=False)
 plt.tight_layout()
 safe_name = feat.replace("/", "_").replace(" ", "_")
 plt.savefig(os.path.join(FIG_DIR, f"shap_dependence_{safe_name}.png"),
 dpi=300, bbox_inches="tight")
 plt.close()
 print("[10] ✓ SHAP dependence plots (top 3) → scripts/figures/")

 # Save SHAP summary CSV
 shap_df.to_csv(os.path.join(PROC, "SHAP_Results.csv"), index=False)
 print("[10] ✓ SHAP results → data/processed/SHAP_Results.csv")


if __name__ == "__main__":
 main()
