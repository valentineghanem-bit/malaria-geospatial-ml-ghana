#!/usr/bin/env python3
"""
02_global_morans.py
Compute Global Moran's I for malaria burden indicators across 260 Ghana districts.
Author: Valentine Golden Ghanem | April 2026
Inputs: data/processed/Ghana_Malaria_260District_MasterDataset.csv
 data/raw/Ghana_New_260_District.geojson
Outputs: data/processed/morans_results.csv
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from libpysal.weights import KNN, Rook
from esda.moran import Moran
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
PERMUTATIONS = 999
K = 8

def load_data():
 gdf = gpd.read_file('data/raw/Ghana_New_260_District.geojson').to_crs('EPSG:32630')
 df = pd.read_csv('data/processed/Ghana_Malaria_260District_MasterDataset.csv')
 gdf['district'] = gdf['DISTRICT'].str.strip()
 return gdf.merge(df, on='district', how='left')

def compute_morans(gdf, indicators, w):
 results = []
 for ind in indicators:
 y = gdf[ind].fillna(gdf[ind].median()).values
 mi = Moran(y, w, permutations=PERMUTATIONS, seed=RANDOM_SEED)
 results.append({'indicator': ind, 'morans_I': round(mi.I, 4),
 'z_score': round(mi.z_norm, 3), 'p_value': round(mi.p_norm, 4)})
 print(f" {ind}: I={mi.I:.4f}, z={mi.z_norm:.3f}, p={mi.p_norm:.4f}")
 return pd.DataFrame(results)

if __name__ == '__main__':
 gdf = load_data()
 w = KNN.from_dataframe(gdf, k=K)
 w.transform = 'r' # row-standardise
 indicators = ['incidence','parasitemia','itn_coverage','water_access','sanitation','u5mr']
 print(f"Global Moran's I (KNN k={K}; {PERMUTATIONS} permutations):")
 results = compute_morans(gdf, indicators, w)
 results.to_csv('data/processed/morans_results.csv', index=False)
 print("\nResults saved to data/processed/morans_results.csv")
