#!/usr/bin/env python3
"""
04_getis_ord.py — Ghana Malaria 260-District Analysis
Getis-Ord Gi* hotspot / coldspot delineation (KNN k=8, 999 permutations).
Significance tiers: 95% (|z|≥1.96), 99% (|z|≥2.576), 99.9% (|z|≥3.291).

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Expected: Priority-1 (99.9% CI) = 38 districts | Total hotspot = 51 | Total coldspot = 42
Outputs : data/processed/Getis_Ord_Results.csv
Tenet 12 (Policy Bridge): districts with Gi* Priority-1 flagged for /policy-bridge.
"""

import os
import pickle
import numpy as np
import pandas as pd
from esda.getisord import G_Local

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(REPO_ROOT, "data", "processed")
SEED = 42


def gi_class(z: float) -> str:
 """Classify district by Gi* z-score tier (two-tailed thresholds)."""
 az = abs(z)
 if z > 0:
 if az >= 3.291:
 return "Hotspot_99.9pct" # Priority-1 for policy targeting
 if az >= 2.576:
 return "Hotspot_99pct"
 if az >= 1.960:
 return "Hotspot_95pct"
 else:
 if az >= 2.576:
 return "Coldspot_99pct"
 if az >= 1.960:
 return "Coldspot_95pct"
 return "Not_Significant"


def main() -> None:
 df = pd.read_csv(os.path.join(PROC, "Ghana_Malaria_260District_MasterDataset.csv"))
 assert len(df) == 260

 inc_col = next(c for c in df.columns if "incidence" in c.lower())
 dist_col = next(c for c in df.columns if "district" in c.lower() or "name" in c.lower())
 y = df[inc_col].values

 with open(os.path.join(PROC, "weights_knn8.pkl"), "rb") as fh:
 w = pickle.load(fh)

 np.random.seed(SEED)
 gi = G_Local(y, w, transform="r", permutations=999, star=True, seed=SEED)

 res = pd.DataFrame({
 "district": df[dist_col].values,
 "Gi_z_score": gi.Zs,
 "Gi_p_sim": gi.p_sim,
 "Gi_Classification": [gi_class(z) for z in gi.Zs],
 "malaria_incidence": y,
 })

 counts = res["Gi_Classification"].value_counts()
 n_hot = sum(counts.get(c, 0) for c in ["Hotspot_99.9pct", "Hotspot_99pct", "Hotspot_95pct"])
 n_cold = sum(counts.get(c, 0) for c in ["Coldspot_99pct", "Coldspot_95pct"])
 p1 = counts.get("Hotspot_99.9pct", 0)

 print("\n[/uq-flag] Gi* Hotspot/Coldspot Summary (KNN k=8, 999 perm.):")
 for cls, n in sorted(counts.items()):
 print(f" {cls:28s}: {n:3d}")
 print(f"\n → Total hotspot districts : {n_hot} (expected 51)")
 print(f" → Priority-1 (99.9% CI) : {p1} (expected 38)")
 print(f" → Total coldspot districts: {n_cold} (expected 42)")
 print(f" → Gi* z-range : [{gi.Zs.min():.3f}, {gi.Zs.max():.3f}]")

 # Policy bridge flag — Tenet 12
 hotspot_names = res[res["Gi_Classification"] == "Hotspot_99.9pct"]["district"].tolist()
 print(f"\n[12] /policy-bridge: Priority-1 districts requiring intervention:")
 for name in hotspot_names[:10]:
 print(f" {name}")
 if len(hotspot_names) > 10:
 print(f" ... and {len(hotspot_names)-10} more (see Getis_Ord_Results.csv)")

 out = os.path.join(PROC, "Getis_Ord_Results.csv")
 res.to_csv(out, index=False)
 print(f"\n[04] ✓ Gi* results → {out}")


if __name__ == "__main__":
 main()
