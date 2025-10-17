# utils/load_real_data.py (Fixed to use API, no manual file needed)

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import requests
import io
from tqdm import tqdm

# Function to fetch real events
def fetch_real_events():
    url = "https://www.datos.gov.co/api/views/sgp4-3e6k/rows.csv?accessType=DOWNLOAD"
    try:
        response = requests.get(url, timeout=100)  # Keep high timeout
        events_df = pd.read_csv(io.StringIO(response.text))
        # Filter Antioquia, sample 500 events
        antioquia_events = events_df[events_df['DEPARTAMENTO'] == 'ANTIOQUIA'][['LONGITUD_CABECERA', 'LATITUD_CABECERA']].rename(columns={'LONGITUD_CABECERA': 'lon', 'LATITUD_CABECERA': 'lat'}).sample(500, random_state=42)
        return gpd.GeoDataFrame(antioquia_events, geometry=[Point(xy) for xy in zip(antioquia_events.lon, antioquia_events.lat)], crs='EPSG:4326')
    except Exception as e:
        print(f"Failed to fetch real data: {e}. Using synthetic events.")
        return gpd.GeoDataFrame({
            'lat': np.random.uniform(6.0, 7.0, 500),
            'lon': np.random.uniform(-75.5, -75.0, 500)
        }, geometry=[Point(xy) for xy in zip(np.random.uniform(-75.5, -75.0, 500), np.random.uniform(6.0, 7.0, 500))], crs='EPSG:4326')

# Generate Terrain Grid (1000 cells)
np.random.seed(42)
n_samples = 1000
terrain_df = pd.DataFrame({
    'lat': np.random.uniform(6.0, 7.0, n_samples),  # Antioquia bounds
    'lon': np.random.uniform(-75.5, -75.0, n_samples),
    'elevation': np.random.uniform(500, 2500, n_samples),  # IGAC-like
    'land_use': np.random.choice(['Forest', 'Agricultural', 'Urban', 'Mining'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'rainfall': np.random.uniform(1500, 4000, n_samples)  # WorldClim-like
})
terrain_gdf = gpd.GeoDataFrame(terrain_df, geometry=[Point(xy) for xy in zip(terrain_df.lon, terrain_df.lat)], crs='EPSG:4326')

# Fetch events
events_gdf = fetch_real_events()

# Project to a metric CRS for accurate distances (EPSG:3116 for Colombia)
terrain_gdf = terrain_gdf.to_crs('EPSG:3116')
events_gdf = events_gdf.to_crs('EPSG:3116')

# Compute min distance
distances = []
for geom in tqdm(terrain_gdf.geometry, desc="Calculating distances"):
    min_dist = events_gdf.geometry.distance(geom).min()
    distances.append(min_dist)
terrain_gdf['dist_to_hist_mine'] = distances

# Add labels
terrain_gdf['label'] = np.random.choice([0, 1], n_samples, p=[0.984, 0.016])  # Imbalanced, ~16 positives

# Reproject back to lat/lon for mapping
terrain_gdf = terrain_gdf.to_crs('EPSG:4326')

# Save
terrain_gdf.to_csv('data/terrain_data.csv', index=False)
terrain_gdf.to_file('data/terrain_data.geojson', driver='GeoJSON')

print(f"Dataset ready! Shape: {terrain_gdf.shape}, Positives: {terrain_gdf['label'].mean():.3f}")
print(terrain_gdf[['lat', 'lon', 'dist_to_hist_mine', 'elevation', 'land_use', 'label']].head())

# import pandas as pd
# import geopandas as gpd
# import numpy as np
# from shapely.geometry import Point
# import requests
# import io
# from tqdm import tqdm  # For progress bar in loop

# # Function to fetch real events
# def fetch_real_events():
#     url = "https://www.datos.gov.co/api/views/sgp4-3e6k/rows.csv?accessType=DOWNLOAD"
#     try:
#         response = requests.get(url, timeout=100)
#         events_df = pd.read_csv(io.StringIO(response.text))
#         # Filter Antioquia, sample 500 events (adjust if needed)
#         antioquia_events = events_df[events_df['DEPARTAMENTO'] == 'ANTIOQUIA'][['LONGITUD_CABECERA', 'LATITUD_CABECERA']].rename(columns={'LONGITUD_CABECERA': 'lon', 'LATITUD_CABECERA': 'lat'}).sample(500, random_state=42)
#         return gpd.GeoDataFrame(antioquia_events, geometry=[Point(xy) for xy in zip(antioquia_events.lon, antioquia_events.lat)], crs='EPSG:4326')
#     except Exception as e:
#         print(f"Failed to fetch real data: {e}. Using synthetic events.")
#         # Fallback: Synthetic event points
#         return gpd.GeoDataFrame({
#             'lat': np.random.uniform(6.0, 7.0, 500),
#             'lon': np.random.uniform(-75.5, -75.0, 500)
#         }, geometry=[Point(xy) for xy in zip(np.random.uniform(-75.5, -75.0, 500), np.random.uniform(6.0, 7.0, 500))], crs='EPSG:4326')

# # Generate Terrain Grid (1000 cells)
# np.random.seed(42)
# n_samples = 1000
# terrain_df = pd.DataFrame({
#     'lat': np.random.uniform(6.0, 7.0, n_samples),  # Antioquia bounds
#     'lon': np.random.uniform(-75.5, -75.0, n_samples),
#     'elevation': np.random.uniform(500, 2500, n_samples),  # IGAC-like
#     'land_use': np.random.choice(['Forest', 'Agricultural', 'Urban', 'Mining'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
#     'rainfall': np.random.uniform(1500, 4000, n_samples)  # WorldClim-like
# })
# terrain_gdf = gpd.GeoDataFrame(terrain_df, geometry=[Point(xy) for xy in zip(terrain_df.lon, terrain_df.lat)], crs='EPSG:4326')

# # Fetch events
# events_gdf = fetch_real_events()

# # Project to a metric CRS for accurate distances (EPSG:3116 for Colombia)
# terrain_gdf = terrain_gdf.to_crs('EPSG:3116')
# events_gdf = events_gdf.to_crs('EPSG:3116')

# # Compute min distance (loop to avoid alignment issues and ensure DataFrame/Series handling)
# distances = []
# for geom in tqdm(terrain_gdf.geometry, desc="Calculating distances"):
#     min_dist = events_gdf.geometry.distance(geom).min()
#     distances.append(min_dist)
# terrain_gdf['dist_to_hist_mine'] = distances

# # Add labels
# terrain_gdf['label'] = np.random.choice([0, 1], n_samples, p=[0.984, 0.016])  # Imbalanced, ~16 positives

# # Reproject back to lat/lon for mapping
# terrain_gdf = terrain_gdf.to_crs('EPSG:4326')

# # Save
# terrain_gdf.to_csv('data/terrain_data.csv', index=False)
# terrain_gdf.to_file('data/terrain_data.geojson', driver='GeoJSON')

# print(f"Dataset ready! Shape: {terrain_gdf.shape}, Positives: {terrain_gdf['label'].mean():.3f}")
# print(terrain_gdf[['lat', 'lon', 'dist_to_hist_mine', 'elevation', 'land_use', 'label']].head())