"""Tiny RAG-style geospatial query helper.

This module provides a very small retrieval layer that converts a limited set of
natural language queries into deterministic geospatial filters applied to the
project's terrain dataset (data/terrain_data.geojson or data/terrain_data.csv).

Limitations & assumptions:
- No external LLM is called. The "language understanding" is simple keyword and
  regex based and intended for short, constrained queries like the example.
- For "near rivers" we look for a `land_use` column containing values like
  'river', 'water', 'stream' or for a `data/rivers.geojson` file if present.
- Country filtering (e.g., 'Colombia') is applied only if a `country` column
  exists in the dataset; otherwise it is ignored.
"""
from __future__ import annotations
import re
import os
import json
from typing import Dict, Any, Tuple
import geopandas as gpd
from shapely.geometry import Point


def _load_gdf(path_data: str = 'data/terrain_data.geojson') -> gpd.GeoDataFrame:
    """Load the terrain GeoDataFrame from geojson or CSV fallback."""
    if os.path.exists(path_data):
        gdf = gpd.read_file(path_data)
    else:
        csvp = path_data.replace('.geojson', '.csv')
        if os.path.exists(csvp):
            df = __import__('pandas').read_csv(csvp)
            if 'geometry' in df.columns:
                gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df.geometry), crs='EPSG:4326')
            else:
                gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs='EPSG:4326')
        else:
            raise FileNotFoundError('No terrain data found at expected paths')
    return gdf


def _find_river_candidates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return subset of gdf that looks like rivers/water based on land_use or keywords.

    If there is a separate `data/rivers.geojson` file, prefer that.
    """
    rivers_path = os.path.join('data', 'rivers.geojson')
    if os.path.exists(rivers_path):
        try:
            return gpd.read_file(rivers_path)
        except Exception:
            pass

    if 'land_use' in gdf.columns:
        mask = gdf['land_use'].astype(str).str.lower().str.contains('river|stream|water|lake|wetland')
        return gdf[mask]

    # no obvious river layer — return empty GeoDataFrame with same CRS
    return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)


def _parse_threshold(q: str) -> float | None:
    """Extract a simple numeric threshold like '> 0.7' or 'above 0.7' from text."""
    m = re.search(r'>\s*([01]?\.?\d+)', q)
    if m:
        try:
            v = float(m.group(1))
            return v
        except Exception:
            return None
    m = re.search(r'above\s*([01]?\.?\d+)', q)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def answer_query(query: str, out_geojson: str = 'data/rag_query_result.geojson') -> Dict[str, Any]:
    """Answer a constrained geospatial query and save results as GeoJSON.

    Returns a dict with keys: 'text' (plain-English answer), 'path' (GeoJSON path or None),
    'n_matches' (int).
    """
    q = query.lower()
    gdf = _load_gdf()

    # Parse risk threshold
    thr = _parse_threshold(q)

    # Country filter (if applicable)
    country = None
    m_country = re.search(r'colombia|colombian', q)
    if m_country and 'country' in gdf.columns:
        country = 'Colombia'

    # 'near rivers' intent
    near_rivers = 'river' in q or 'rivers' in q or 'near river' in q or 'near rivers' in q or 'near water' in q

    # Start filtering candidates by threshold & country
    cand = gdf.copy()
    if country is not None:
        cand = cand[cand['country'].astype(str).str.contains(country, case=False, na=False)]

    if thr is not None and 'risk' in cand.columns:
        cand = cand[cand['risk'].astype(float) > float(thr)]

    # If rivers requested, perform spatial intersection / proximity
    if near_rivers:
        rivers = _find_river_candidates(gdf)
        if len(rivers) == 0:
            # can't do spatial proximity without river geometry; report empty
            cand = cand.iloc[0:0]
        else:
            # ensure same CRS for metric distances
            try:
                rivers_m = rivers.to_crs(epsg=3857)
                cand_m = cand.to_crs(epsg=3857)
                # find points within 500m of any river geometry
                rivers_union = rivers_m.unary_union
                cand['near_river'] = cand_m.geometry.apply(lambda geom: geom.distance(rivers_union) <= 500)
                cand = cand[cand['near_river'] == True]
            except Exception:
                # fallback conservative: empty result
                cand = cand.iloc[0:0]

    n = len(cand)
    if n > 0:
        # save GeoJSON result
        try:
            os.makedirs(os.path.dirname(out_geojson) or '.', exist_ok=True)
            cand.to_file(out_geojson, driver='GeoJSON')
            path = out_geojson
        except Exception:
            path = None
    else:
        path = None

    # Craft a short English answer
    parts = []
    if thr is not None:
        parts.append(f'Predicted risk > {thr}')
    if country is not None:
        parts.append(f'in {country}')
    if near_rivers:
        parts.append('near rivers')
    head = ' and '.join(parts) if parts else 'matching your criteria'
    text = f'Found {n} region(s) {head}.'

    return {'text': text, 'path': path, 'n_matches': n}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--query', type=str, required=True)
    args = p.parse_args()
    out = answer_query(args.query)
    print(out['text'])