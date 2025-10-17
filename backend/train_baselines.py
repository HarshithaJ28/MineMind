"""
backend/train_baselines.py

Train baseline classifiers (LogisticRegression, RandomForest) using preprocessed artifacts,
compute cross-validated ROC-AUC and PR-AUC, evaluate on test set, produce ROC/PR/Calibration
plots and save best model and metrics to disk.

Usage:
    python backend/train_baselines.py

"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')
REPORTS_DIR = os.path.join(ROOT, 'reports')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_artifacts():
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()
    return X_train, X_test, y_train, y_test


def fit_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'logreg': LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    }

    # Hyperparameter tuning for RandomForest via RandomizedSearchCV
    rf_param_dist = {
        'n_estimators': [100, 200, 400, 800],
        'max_depth': [None, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.3, 0.5]
    }
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    rnd_search = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=20, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
    try:
        rnd_search.fit(X_train, y_train)
        best_rf = rnd_search.best_estimator_
        models['rf'] = best_rf
        rf_search_info = {'best_params': rnd_search.best_params_, 'best_score': float(rnd_search.best_score_)}
    except Exception:
        # fallback to default RF if tuning fails
        models['rf'] = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_search_info = {'best_params': None, 'best_score': None}

    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
        cv_ap = cross_val_score(model, X_train, y_train, cv=skf, scoring='average_precision', n_jobs=-1)
        results[name] = {'cv_roc_auc_mean': float(np.mean(cv_auc)), 'cv_roc_auc_std': float(np.std(cv_auc)),
                         'cv_ap_mean': float(np.mean(cv_ap)), 'cv_ap_std': float(np.std(cv_ap))}

    # fit on full train and evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)
        results[name].update({'test_roc_auc': float(roc), 'test_ap': float(ap)})
        if roc > best_score:
            best_score = roc
            best_model = (name, model)


    # save best (uncalibrated) model
    best_name, best_est = best_model
    uncalibrated_path = os.path.join(MODELS_DIR, f'baseline_best_{best_name}.pkl')
    joblib.dump(best_est, uncalibrated_path)

    # Calibration: try sigmoid (Platt) and isotonic if possible, choose by Brier score on test set
    calibration_results = {}
    try:
        methods = ['sigmoid', 'isotonic']
        calibrated_candidates = {}
        for method in methods:
            try:
                calib = CalibratedClassifierCV(best_est, method=method, cv=5)
                calib.fit(X_train, y_train)
                probs_cal = calib.predict_proba(X_test)[:, 1]
                brier = brier_score_loss(y_test, probs_cal)
                calibrated_candidates[method] = {'model': calib, 'brier': float(brier), 'probs': probs_cal}
            except Exception:
                # method failed (e.g., isotonic needs more data)
                calibrated_candidates[method] = {'model': None, 'brier': None, 'probs': None}

        # pick best calibration (lowest brier)
        best_calib_method = None
        best_brier = None
        best_calib = None
        for m, info in calibrated_candidates.items():
            if info['brier'] is not None and (best_brier is None or info['brier'] < best_brier):
                best_brier = info['brier']
                best_calib_method = m
                best_calib = info['model']

        if best_calib is not None:
            calibrated_path = os.path.join(MODELS_DIR, f'baseline_best_{best_name}_calibrated_{best_calib_method}.pkl')
            joblib.dump(best_calib, calibrated_path)
            # use calibrated probs for final reporting
            final_probs = best_calib.predict_proba(X_test)[:, 1]
            final_brier = float(brier_score_loss(y_test, final_probs))
            calibration_results = {'method': best_calib_method, 'brier': final_brier, 'calibrated_path': calibrated_path}
        else:
            # no calibration succeeded
            final_probs = best_est.predict_proba(X_test)[:, 1]
            calibration_results = {'method': None, 'brier': float(brier_score_loss(y_test, final_probs)), 'calibrated_path': None}
    except Exception:
        # fallback
        final_probs = best_est.predict_proba(X_test)[:, 1]
        calibration_results = {'method': None, 'brier': float(brier_score_loss(y_test, final_probs)), 'calibrated_path': None}

    # recompute plots using final_probs (calibrated if available)
    fpr, tpr, _ = roc_curve(y_test, final_probs)
    precision, recall, _ = precision_recall_curve(y_test, final_probs)

    # produce plots for best model
    probs = best_est.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_score(y_test, final_probs):.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_path = os.path.join(REPORTS_DIR, 'roc.png')
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR (AP={average_precision_score(y_test, final_probs):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    pr_path = os.path.join(REPORTS_DIR, 'pr.png')
    plt.savefig(pr_path, bbox_inches='tight')
    plt.close()

    # Calibration plot
    prob_true, prob_pred = calibration_curve(y_test, final_probs, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical probability')
    plt.title('Calibration curve')
    plt.legend()
    cal_path = os.path.join(REPORTS_DIR, 'calibration.png')
    plt.savefig(cal_path, bbox_inches='tight')
    plt.close()

    # Save metrics and metadata
    report = {
        'best_model': best_name,
        'best_score_test_roc_auc': float(best_score),
        'models': results,
        'plots': {'roc': roc_path, 'pr': pr_path, 'calibration': cal_path},
        'rf_search': rf_search_info,
        'uncalibrated_model_path': uncalibrated_path,
        'calibration': calibration_results
    }
    with open(os.path.join(REPORTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print('Training complete. Best model:', best_name)
    print('Reports saved to', REPORTS_DIR)


def main():
    X_train, X_test, y_train, y_test = load_artifacts()
    fit_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
