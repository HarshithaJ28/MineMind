"""
backend/preprocess.py

Purpose: Robust preprocessing for MineRiskMapper Phase 1.

Features:
- Load geojson or CSV from data/
- Basic feature engineering (one-hot land_use, log distance)
- Imputation and scaling
- Stratified train/test split
- Save artifacts: X_train/X_test/y_train/y_test, scaler, feature_columns

Usage:
    python backend/preprocess.py

"""
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_terrain(path_geojson=None, path_csv=None):
    if path_geojson and os.path.exists(path_geojson):
        gdf = gpd.read_file(path_geojson)
    elif path_csv and os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
        if 'geometry' in df.columns:
            try:
                gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df.geometry), crs='EPSG:4326')
            except Exception:
                gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs='EPSG:4326')
        else:
            gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs='EPSG:4326')
    else:
        raise FileNotFoundError('Provide either path_geojson or path_csv that exists')
    # ensure lat/lon
    if 'lat' not in gdf.columns or 'lon' not in gdf.columns:
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
    return gdf


def basic_feature_engineering(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    df = gdf.copy()
    # ensure expected columns exist
    for col in ['dist_to_hist_mine', 'elevation', 'rainfall', 'land_use']:
        if col not in df.columns:
            df[col] = np.nan

    # log transform distances to reduce skew
    df['log_dist'] = np.log1p(df['dist_to_hist_mine'].fillna(df['dist_to_hist_mine'].median()))

    # compute derived spatial features: slope and aspect using local plane fit in metric CRS
    try:
        metric = df.to_crs(epsg=3857)
        coords = np.vstack([metric.geometry.x.values, metric.geometry.y.values]).T
        elev = metric['elevation'].fillna(metric['elevation'].median()).values
        # For each point, fit plane z = ax + by + c on k nearest neighbors
        from sklearn.neighbors import KDTree
        k = min(12, len(metric) - 1)
        tree = KDTree(coords)
        neigh_idx = tree.query(coords, k=k + 1, return_distance=False)[:, 1:]
        slopes = np.zeros(len(metric))
        aspects = np.zeros(len(metric))
        for i in range(len(metric)):
            ids = neigh_idx[i]
            X = np.column_stack([coords[ids, 0], coords[ids, 1], np.ones(len(ids))])
            z = elev[ids]
            # solve plane params via least squares
            try:
                a, b, c = np.linalg.lstsq(X, z, rcond=None)[0]
                # gradient vector is [a, b]
                slope = np.sqrt(a * a + b * b)
                aspect = (np.degrees(np.arctan2(b, a)) + 360) % 360
            except Exception:
                slope = 0.0
                aspect = 0.0
            slopes[i] = slope
            aspects[i] = aspect
        df['slope'] = slopes
        df['aspect'] = aspects
    except Exception:
        df['slope'] = 0.0
        df['aspect'] = 0.0

    # neighborhood vegetation fraction: look for common veg column names
    veg_cols = [c for c in df.columns if 'veg' in c.lower() or 'ndvi' in c.lower() or 'veget' in c.lower()]
    if len(veg_cols) > 0:
        veg_col = veg_cols[0]
        try:
            metric = df.to_crs(epsg=3857)
            coords = np.vstack([metric.geometry.x.values, metric.geometry.y.values]).T
            tree = KDTree(coords)
            # compute fraction of neighbors above median veg
            veg_vals = df[veg_col].fillna(df[veg_col].median()).values
            neigh = tree.query(coords, k=8, return_distance=False)[:, 1:]
            veg_frac = np.array([veg_vals[ids].mean() for ids in neigh])
            df['veg_frac'] = veg_frac
        except Exception:
            df['veg_frac'] = df[veg_col].fillna(df[veg_col].median())
    else:
        df['veg_frac'] = 0.0

    # local point density (per km^2) using 1 km radius
    try:
        metric = df.to_crs(epsg=3857)
        coords = np.vstack([metric.geometry.x.values, metric.geometry.y.values]).T
        tree = KDTree(coords)
        radius = 1000.0
        counts = tree.query_radius(coords, r=radius, count_only=True)
        area_km2 = np.pi * (radius ** 2) / 1e6
        df['local_density'] = counts / area_km2
    except Exception:
        df['local_density'] = 0.0

    # one-hot land use
    land_dummies = pd.get_dummies(df['land_use'].fillna('Unknown'), prefix='land')

    features = pd.concat([
        df[['elevation', 'rainfall', 'log_dist', 'slope', 'aspect', 'veg_frac', 'local_density']].fillna(-999),
        land_dummies
    ], axis=1)

    # fill nan with sentinel for now
    features = features.fillna(-999)
    return features


def stratified_split(X, y, test_size=0.2, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def scale_and_save(X_train, X_test, out_dir=MODELS_DIR):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    # save scaler and feature columns
    import joblib
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))
    with open(os.path.join(out_dir, 'feature_columns.json'), 'w') as f:
        json.dump(list(X_train.columns), f)
    return X_train_s, X_test_s


def save_artifacts(X_train, X_test, y_train, y_test, out_dir=DATA_DIR):
    X_train.to_csv(os.path.join(out_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)


def main():
    gdf = load_terrain(path_geojson=os.path.join(DATA_DIR, 'terrain_data.geojson'), path_csv=os.path.join(DATA_DIR, 'terrain_data.csv'))
    features = basic_feature_engineering(gdf)
    if 'label' not in gdf.columns:
        raise ValueError('Input data must contain a `label` column for supervised training')
    y = gdf['label'].astype(int)
    X_train, X_test, y_train, y_test = stratified_split(features, y)
    X_train_s, X_test_s = scale_and_save(X_train, X_test)
    save_artifacts(X_train_s, X_test_s, y_train, y_test)
    print('Preprocessing complete. Artifacts saved to data/ and models/')


if __name__ == '__main__':
    main()
