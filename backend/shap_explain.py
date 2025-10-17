"""Compute and persist SHAP values for the dataset and model.

Usage:
  python backend/shap_explain.py

Saves artifacts to `models/`:
 - shap_values.npy  (n_samples x n_features)
 - shap_features.json (list of feature names)
 - shap_top_contribs.csv (index,top_pos,top_neg)
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import joblib


def load_features(features_path='models/feature_columns.json'):
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            return json.load(f)
    return None


def compute_shap_for_dataset(model_path=None, data_path='data/terrain_data.csv', features_path='models/feature_columns.json', out_dir='models', sample_size: int | None = None):
    try:
        import shap
    except Exception as e:
        raise RuntimeError('shap is required for this script. pip install shap') from e

    if model_path is None:
        # try common model filenames
        candidates = ['models/baseline_best_logreg.pkl', 'models/risk_model.pkl', 'models/baseline_best_logreg_calibrated_isotonic.pkl']
        model_path = next((c for c in candidates if os.path.exists(c)), None)
    if model_path is None:
        raise FileNotFoundError('No model found in models/. Please train a model first.')

    model = joblib.load(model_path)

    # load data
    df = pd.read_csv(data_path)
    features = load_features(features_path)
    if features is None:
        # fallback: use heuristics
        features = [c for c in df.columns if c not in ('lat', 'lon', 'geometry', 'label')]

    # build design matrix with dummies and align to saved features if present
    X = pd.get_dummies(df[[f for f in features if f in df.columns] + (['land_use'] if 'land_use' in df.columns and 'land_use' not in features else [])])
    # align
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            feat_names = json.load(f)
        for c in feat_names:
            if c not in X.columns:
                X[c] = 0
        X = X[feat_names]
    else:
        feat_names = list(X.columns)

    # Choose explainer based on model type
    try:
        # Tree models benefit from TreeExplainer
        expl = shap.TreeExplainer(model)
    except Exception:
        expl = shap.Explainer(model, X)

    # optionally sample for approximate SHAP using shap.sample if available
    if sample_size is not None and sample_size > 0 and sample_size < len(X):
        try:
            # shap.sample can produce a representative sample for SHAP
            Xs = shap.sample(X, sample_size, random_state=42)
        except Exception:
            # fallback to pandas sampling
            if 'label' in df.columns:
                pos = df[df.label == 1]
                neg = df[df.label == 0]
                keep_pos = int(min(len(pos), max(1, sample_size // 10)))
                keep_neg = sample_size - keep_pos
                sample_idx = pd.concat([pos.sample(n=keep_pos, replace=False, random_state=42), neg.sample(n=keep_neg, replace=False, random_state=42)]).index
                Xs = X.loc[sample_idx]
            else:
                Xs = X.sample(n=sample_size, random_state=42)
        print('Computing SHAP values on sample', Xs.shape)
        shap_vals = expl.shap_values(Xs)
    else:
        print('Computing SHAP values for', X.shape)
        shap_vals = expl.shap_values(X)

    # shap.TreeExplainer returns list for binary classification (two arrays), else array
    if isinstance(shap_vals, list) and len(shap_vals) >= 2:
        # for classifiers, shap for the positive class is typically index 1
        shap_arr = np.array(shap_vals[1])
    else:
        shap_arr = np.array(shap_vals)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'shap_values.npy'), shap_arr)
    with open(os.path.join(out_dir, 'shap_features.json'), 'w') as f:
        json.dump(feat_names, f)

    # compute and cache per-feature mean absolute SHAP summary for fast UI loading
    try:
        mean_abs = np.mean(np.abs(shap_arr), axis=0)
        summary = {feat_names[i]: float(mean_abs[i]) for i in range(len(feat_names))}
        # sort descending
        summary_sorted = dict(sorted(summary.items(), key=lambda kv: kv[1], reverse=True))
        with open(os.path.join(out_dir, 'shap_summary.json'), 'w') as f:
            json.dump(summary_sorted, f, indent=2)
        pd.DataFrame(list(summary_sorted.items()), columns=['feature', 'mean_abs_shap']).to_csv(os.path.join(out_dir, 'shap_summary.csv'), index=False)
        print('Saved SHAP summary to', out_dir)
    except Exception as e:
        print('Failed to compute SHAP summary:', e)

    # produce simple per-row top contributors CSV
    top_rows = []
    for i, row in enumerate(shap_arr):
        abs_row = np.abs(row)
        # top 3 positive and top 3 negative
        idx_pos = list(np.argsort(-row)[:3])
        idx_neg = list(np.argsort(row)[:3])
        top_pos = ';'.join([f"{feat_names[j]}:{row[j]:.4f}" for j in idx_pos])
        top_neg = ';'.join([f"{feat_names[j]}:{row[j]:.4f}" for j in idx_neg])
        top_rows.append({'index': i, 'top_pos': top_pos, 'top_neg': top_neg})

    pd.DataFrame(top_rows).to_csv(os.path.join(out_dir, 'shap_top_contribs.csv'), index=False)
    print('Saved SHAP artifacts to', out_dir)


if __name__ == '__main__':
    # small CLI to pass sample size
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=None, help='If set, compute SHAP on a small sample for speed')
    args = parser.parse_args()
    compute_shap_for_dataset(sample_size=args.sample_size)
