"""Simple Active Learning utilities: query-by-uncertainty and incremental retrain.

Usage examples:
  python backend/active_learning.py query --n 20
  python backend/active_learning.py retrain

This is intentionally lightweight: it uses uncertainty (per-sample std) and allows saving queried indices to `models/active_queries.json`.
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib


def load_data(data_path='data/terrain_data.csv', labels_delta='data/labels_delta.csv'):
    df = pd.read_csv(data_path)
    # merge label deltas if available (do not overwrite original data file)
    if os.path.exists(labels_delta):
        delta = pd.read_csv(labels_delta)
        # expect columns: index,label (index refers to original df index)
        for _, row in delta.iterrows():
            idx = int(row['index'])
            if 0 <= idx < len(df):
                df.at[idx, 'label'] = int(row['label'])
    return df


def load_model(model_path='models/risk_model.pkl'):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def compute_uncertainty_for_dataset(model, df, features):
    # Build X same as training
    X = pd.get_dummies(df[features + (['land_use'] if 'land_use' in df.columns else [])])
    # align to model features if present
    feat_names = getattr(model, 'feature_names_in_', None)
    if feat_names is not None:
        for c in feat_names:
            if c not in X.columns:
                X[c] = 0
        X = X[feat_names]
    # per-tree std if RandomForest: decide whether to pass DataFrame or numpy depending on estimator feature names
    try:
        estimators = getattr(model, 'estimators_', [])
        use_df_for_trees = all(hasattr(est, 'feature_names_in_') for est in estimators)
        if hasattr(model, 'feature_names_in_'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict_proba(X.to_numpy())[:, 1]

        # Call individual trees with numpy arrays (avoid feature-names warning for trees)
        X_np = X.to_numpy()
        all_tree_probs = np.stack([est.predict_proba(X_np)[:, 1] for est in estimators], axis=1)
        uncertainty = all_tree_probs.std(axis=1)
    except Exception:
        probs = model.predict_proba(X)[:, 1] if hasattr(model, 'feature_names_in_') else model.predict_proba(X.to_numpy())[:, 1]
        ent = - (probs * np.log(probs + 1e-9) + (1 - probs) * np.log(1 - probs + 1e-9))
        uncertainty = ent / (ent.max() + 1e-9)
    return uncertainty


def query_by_uncertainty(n=20, out_json='models/active_queries.json'):
    df = load_data()
    model = load_model()
    if model is None:
        raise FileNotFoundError('No model found at models/risk_model.pkl')
    # attempt to infer features from model
    features = list(getattr(model, 'feature_names_in_', []))
    if len(features) == 0:
        # fallback heuristics
        features = [c for c in df.columns if c not in ('lat','lon','geometry','label')]

    uncertainty = compute_uncertainty_for_dataset(model, df, features)
    idxs = np.argsort(-uncertainty)[:n]
    queries = [{'index': int(i), 'uncertainty': float(uncertainty[int(i)])} for i in idxs]
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'queries': queries}, f, indent=2)
    print(f'Saved top-{n} uncertain sample indices to {out_json}')
    return queries


def retrain_with_labels(data_path='data/terrain_data.csv', model_path='models/risk_model.pkl', features_override=None):
    # load merged view (original + label deltas)
    df = load_data(data_path)
    if 'label' not in df.columns:
        raise ValueError('No label column in data; cannot retrain')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    # construct features
    if features_override is None:
        features = [c for c in df.columns if c not in ('lat','lon','geometry','label')]
    else:
        features = features_override

    X = pd.get_dummies(df[features + (['land_use'] if 'land_use' in df.columns else [])])
    y = df['label']
    feat_names = list(X.columns)

    # choose training strategy: if many new labels prefer incremental (SGD), else full retrain (RF)
    # Heuristic: if dataset < 2000 rows use full RandomForest, else try SGD incremental
    if len(df) < 2000:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        # sklearn estimators accept numpy arrays for fit; keep using numpy here
        clf.fit(X.to_numpy(), y.to_numpy())
    else:
        # SGDClassifier supports partial_fit; we create a simple pipeline-like estimator
        clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        # partial_fit needs classes
        classes = np.unique(y)
        # train in one go via partial_fit
        clf.partial_fit(X.to_numpy(), y.to_numpy(), classes=classes)

    joblib.dump(clf, model_path)
    # save feature names
    os.makedirs(os.path.dirname('models/feature_columns.json'), exist_ok=True)
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feat_names, f)
    print('Retrained and saved model to', model_path)
    return clf


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    q = sub.add_parser('query')
    q.add_argument('--n', type=int, default=20)
    r = sub.add_parser('retrain')
    args = parser.parse_args()
    if args.cmd == 'query':
        query_by_uncertainty(n=args.n)
    elif args.cmd == 'retrain':
        retrain_with_labels()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
