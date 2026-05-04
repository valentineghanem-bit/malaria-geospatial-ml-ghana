#!/usr/bin/env python3
"""
08_random_forest.py — Ghana Malaria 260-District Analysis
Random Forest classifier with LODO spatial cross-validation.

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Outputs: data/processed/rf_results.csv, data/models/rf_model.pkl
Random seed: 42 (Tenet 8)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, brier_score_loss

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(REPO_ROOT, "data", "processed")
MODELS = os.path.join(REPO_ROOT, "data", "models")
os.makedirs(MODELS, exist_ok=True)

SEED = 42
RF_PARAMS = {
 "n_estimators": 500,
 "max_depth": 6,
 "min_samples_leaf": 5,
 "max_features": "sqrt",
 "random_state": SEED,
 "n_jobs": -1,
}


def main() -> None:
 X = pd.read_csv(os.path.join(PROC, "feature_matrix.csv")).values
 targets = pd.read_csv(os.path.join(PROC, "target_vector.csv"))
 y = targets["high_burden_binary"].values

 master = pd.read_csv(os.path.join(PROC, "Ghana_Malaria_260District_MasterDataset.csv"))
 region_col = next((c for c in master.columns if "region" in c.lower()), None)
 groups = master[region_col].values if region_col else np.arange(len(y))

 logo = LeaveOneGroupOut()
 aucs, briers = [], []
 for train_idx, test_idx in logo.split(X, y, groups):
 rf = RandomForestClassifier(**RF_PARAMS)
 rf.fit(X[train_idx], y[train_idx])
 prob = rf.predict_proba(X[test_idx])[:, 1]
 if len(np.unique(y[test_idx])) > 1:
 aucs.append(roc_auc_score(y[test_idx], prob))
 briers.append(brier_score_loss(y[test_idx], prob))

 print(f"\n[/uq-flag] Random Forest LODO-CV:")
 print(f" AUC-ROC: {np.mean(aucs):.4f} (SD={np.std(aucs):.4f})")
 print(f" Brier : {np.mean(briers):.4f}")
 print(f" N folds: {len(aucs)}")

 # Final model on full data
 np.random.seed(SEED)
 final_rf = RandomForestClassifier(**RF_PARAMS)
 final_rf.fit(X, y)
 with open(os.path.join(MODELS, "rf_model.pkl"), "wb") as fh:
 pickle.dump(final_rf, fh)

 pd.DataFrame([{
 "model": "RandomForest",
 "mean_AUC": round(float(np.mean(aucs)), 4),
 "sd_AUC": round(float(np.std(aucs)), 4),
 "mean_Brier": round(float(np.mean(briers)), 4),
 "n_folds": len(aucs),
 "random_seed": SEED,
 }]).to_csv(os.path.join(PROC, "rf_results.csv"), index=False)
 print("[08] ✓ RF model and results saved.")


if __name__ == "__main__":
 main()
