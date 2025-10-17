"""
backend/operating_points.py

Compute operating point metrics (precision, recall, FPR) across thresholds
using the (calibrated) best model and save results to reports/operating_points.csv.

Usage:
    python backend/operating_points.py
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS_DIR = os.path.join(ROOT, 'reports')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(REPORTS_DIR, exist_ok=True)


def find_calibrated_model():
    metrics_path = os.path.join(REPORTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            calib = metrics.get('calibration', {})
            path = calib.get('calibrated_path')
            if path and os.path.exists(path):
                return path
        except Exception:
            pass
    # fallback: try to find any calibrated file
    for fname in os.listdir(MODELS_DIR):
        if 'calibrated' in fname and fname.endswith('.pkl'):
            return os.path.join(MODELS_DIR, fname)
    # fallback: best uncalibrated
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith('baseline_best_') and fname.endswith('.pkl'):
            return os.path.join(MODELS_DIR, fname)
    return None


def load_test_data():
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()
    return X_test, y_test


def compute_operating_points(model_path, X_test, y_test, out_csv):
    model = joblib.load(model_path)
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision = float(precision_score(y_test, preds, zero_division=0))
        recall = float(recall_score(y_test, preds, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        rows.append({'threshold': float(t), 'precision': precision, 'recall': recall, 'fpr': fpr, 'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def main():
    model_path = find_calibrated_model()
    if model_path is None:
        print('No model found in models/. Run training first.')
        return
    X_test, y_test = load_test_data()
    out_csv = os.path.join(REPORTS_DIR, 'operating_points.csv')
    df = compute_operating_points(model_path, X_test, y_test, out_csv)
    print(f'Operating points saved to {out_csv}. Sample:')
    print(df.head())


if __name__ == '__main__':
    main()
