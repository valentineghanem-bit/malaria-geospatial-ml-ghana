#!/usr/bin/env python3
"""
07_xgboost_model.py
XGBoost district-level high malaria burden prediction with LODO spatial cross-validation.
Author: Valentine Golden Ghanem | April 2026
Inputs: data/processed/Ghana_Malaria_260District_MasterDataset.csv
Outputs: data/processed/xgb_lodo_results.csv
 data/processed/xgb_model_metrics.csv
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
XGB_PARAMS = {
 'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05,
 'subsample': 0.8, 'colsample_bytree': 0.8,
 'scale_pos_weight': 1, 'use_label_encoder': False,
 'eval_metric': 'logloss', 'random_state': RANDOM_SEED
}
FEATURES = ['itn_coverage','parasitemia','water_access','sanitation',
 'u5mr','mort_rate','pop_density']

def run_lodo_cv(df):
 """Leave-one-district-out spatial cross-validation."""
 X = StandardScaler().fit_transform(df[FEATURES])
 y = df['high_burden'].values
 districts = df['district'].values
 aucs, briers = [], []
 for i, dist in enumerate(districts):
 train_mask = districts != dist
 X_tr, y_tr = X[train_mask], y[train_mask]
 X_te, y_te = X[~train_mask], y[~train_mask]
 model = XGBClassifier(**XGB_PARAMS)
 model.fit(X_tr, y_tr)
 prob = model.predict_proba(X_te)[:, 1]
 if len(np.unique(y_te)) > 1:
 aucs.append(roc_auc_score(y_te, prob))
 briers.append(brier_score_loss(y_te, prob))
 return np.mean(aucs), np.std(aucs), np.mean(briers)

if __name__ == '__main__':
 df = pd.read_csv('data/processed/Ghana_Malaria_260District_MasterDataset.csv')
 print(f"Running LODO-CV across {len(df)} districts...")
 mean_auc, std_auc, mean_brier = run_lodo_cv(df)
 ci_lo, ci_hi = mean_auc - 1.96*std_auc, mean_auc + 1.96*std_auc
 print(f"XGBoost LODO-CV: AUC-ROC={mean_auc:.3f} (95% CI: {ci_lo:.3f}–{ci_hi:.3f}), Brier={mean_brier:.3f}")
 pd.DataFrame([{'model':'XGBoost','auc_mean':round(mean_auc,3),'auc_std':round(std_auc,3),
 'ci_lo':round(ci_lo,3),'ci_hi':round(ci_hi,3),'brier':round(mean_brier,3)}]
 ).to_csv('data/processed/xgb_model_metrics.csv', index=False)
