#!/usr/bin/env python3
"""
09_cart_logistic.py — Ghana Malaria 260-District Analysis
CART decision tree and Logistic Regression with LODO spatial CV.
Provides interpretable baseline comparators for XGBoost and RF.

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Outputs: data/processed/cart_logistic_results.csv, data/models/cart_model.pkl
Random seed: 42 (Tenet 8)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(REPO_ROOT, "data", "processed")
MODELS = os.path.join(REPO_ROOT, "data", "models")
os.makedirs(MODELS, exist_ok=True)

SEED = 42


def lodo_evaluate(model_template, X: np.ndarray, y: np.ndarray,
 groups: np.ndarray, name: str) -> dict:
 """Run LODO-CV for any sklearn-compatible model."""
 logo = LeaveOneGroupOut()
 aucs, briers = [], []
 for train_idx, test_idx in logo.split(X, y, groups):
 m = type(model_template)(**model_template.get_params())
 m.fit(X[train_idx], y[train_idx])
 prob = m.predict_proba(X[test_idx])[:, 1]
 if len(np.unique(y[test_idx])) > 1:
 aucs.append(roc_auc_score(y[test_idx], prob))
 briers.append(brier_score_loss(y[test_idx], prob))
 print(f"[09] {name}: AUC={np.mean(aucs):.4f} (SD={np.std(aucs):.4f}) "
 f"Brier={np.mean(briers):.4f}")
 return {
 "model": name, "mean_AUC": round(float(np.mean(aucs)), 4),
 "sd_AUC": round(float(np.std(aucs)), 4),
 "mean_Brier": round(float(np.mean(briers)), 4),
 "n_folds": len(aucs),
 }


def main() -> None:
 X = pd.read_csv(os.path.join(PROC, "feature_matrix.csv")).values
 feat_names = pd.read_csv(os.path.join(PROC, "feature_matrix.csv")).columns.tolist()
 targets = pd.read_csv(os.path.join(PROC, "target_vector.csv"))
 y = targets["high_burden_binary"].values

 master = pd.read_csv(os.path.join(PROC, "Ghana_Malaria_260District_MasterDataset.csv"))
 region_col = next((c for c in master.columns if "region" in c.lower()), None)
 groups = master[region_col].values if region_col else np.arange(len(y))

 print("[09] CART and Logistic Regression LODO-CV:")

 # CART
 cart = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
 random_state=SEED)
 cart_res = lodo_evaluate(cart, X, y, groups, "CART")
 cart.fit(X, y) # Final model
 with open(os.path.join(MODELS, "cart_model.pkl"), "wb") as fh:
 pickle.dump(cart, fh)
 print(f"\n[09] CART decision rules (depth≤4):")
 print(export_text(cart, feature_names=feat_names, max_depth=3))

 # Logistic Regression (L2 penalty, scaled)
 lr_pipe = Pipeline([
 ("scaler", StandardScaler()),
 ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
 ])
 lr_res = lodo_evaluate(lr_pipe, X, y, groups, "LogisticRegression_L2")

 # Save comparison table
 results = pd.DataFrame([cart_res, lr_res])
 results.to_csv(os.path.join(PROC, "cart_logistic_results.csv"), index=False)
 print(f"\n[09] ✓ Results → data/processed/cart_logistic_results.csv")


if __name__ == "__main__":
 main()
