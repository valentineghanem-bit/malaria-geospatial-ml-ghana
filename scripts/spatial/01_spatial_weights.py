#!/usr/bin/env python3
"""
01_spatial_weights.py — Ghana Malaria 260-District Analysis
Construct KNN (k=8) and Rook contiguity weight matrices.

Author : Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic
Date : April 2026
Inputs : data/raw/Ghana_New_260_District.geojson
Outputs: data/processed/weights_knn8.pkl, data/processed/weights_rook.pkl
Fail-Fast Gate: Syntax OK | Logic OK | Epi context OK | PEP8 OK
"""

import os
import pickle

import geopandas as gpd
from libpysal.weights import KNN, Rook

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def build_knn(gdf: gpd.GeoDataFrame, k: int = 8):
 """
 KNN (k=8) row-standardised weight matrix.
 Used for: Global Moran's I, Getis-Ord Gi*.
 Justification: k=8 captures typical district neighbourhood density in Ghana.
 """
 gdf_proj = gdf.to_crs(epsg=32630) # UTM Zone 30N — metre-unit centroids
 w = KNN.from_dataframe(gdf_proj, k=k)
 w.transform = "r"
 return w


def build_rook(gdf: gpd.GeoDataFrame):
 """
 Rook contiguity weight matrix.
 Used for: Bivariate LISA (shared boundary edges; excludes diagonal/corner adjacency).
 """
 w = Rook.from_dataframe(gdf)
 w.transform = "r"
 return w


def main() -> None:
 geojson = os.path.join(RAW_DIR, "Ghana_New_260_District.geojson")
 if not os.path.exists(geojson):
 raise FileNotFoundError(
 f"GeoJSON missing: {geojson}\n"
 "Download from Ghana Statistical Service (2021) → data/raw/\n"
 "URL: https://www.statsghana.gov.gh"
 )

 gdf = gpd.read_file(geojson)
 assert len(gdf) == 260, f"Expected 260 districts, got {len(gdf)}"
 print(f"[01] Loaded {len(gdf)} districts | CRS: {gdf.crs}")

 w_knn = build_knn(gdf, k=8)
 knn_path = os.path.join(OUT_DIR, "weights_knn8.pkl")
 with open(knn_path, "wb") as fh:
 pickle.dump(w_knn, fh)
 print(f"[01] KNN(k=8) → {knn_path}")
 print(f" Mean neighbours: {w_knn.mean_neighbors:.2f} | Islands: {w_knn.islands}")

 w_rook = build_rook(gdf)
 rook_path = os.path.join(OUT_DIR, "weights_rook.pkl")
 with open(rook_path, "wb") as fh:
 pickle.dump(w_rook, fh)
 print(f"[01] Rook → {rook_path}")
 print(f" Min neighbours: {w_rook.min_neighbors} | Max: {w_rook.max_neighbors}")

 assert w_knn.n == 260 and w_rook.n == 260
 print("[01] ✓ Spatial weight matrices validated.")


if __name__ == "__main__":
 main()
