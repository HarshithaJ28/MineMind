"""
backend/resampling_experiment.py

Apply SMOTE resampling to the training set, train LogisticRegression and RandomForest,
evaluate on the test set, and save comparison metrics and plots to reports/.

Usage:
    python backend/resampling_experiment.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')
REPORTS_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()
    return X_train, X_test, y_train, y_test


def try_import_smote():
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE
    except Exception as e:
        print('imblearn not installed or failed to import SMOTE:', e)
        return None


def train_and_eval(X_train, y_train, X_test, y_test, name_suffix=''):
    results = {}
    # Logistic Regression
    log = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    log.fit(X_train, y_train)
    probs_log = log.predict_proba(X_test)[:, 1]
    results['logreg' + name_suffix] = {
        'test_roc_auc': float(roc_auc_score(y_test, probs_log)),
        'test_ap': float(average_precision_score(y_test, probs_log)),
        'brier': float(brier_score_loss(y_test, probs_log))
    }

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    probs_rf = rf.predict_proba(X_test)[:, 1]
    results['rf' + name_suffix] = {
        'test_roc_auc': float(roc_auc_score(y_test, probs_rf)),
        'test_ap': float(average_precision_score(y_test, probs_rf)),
        'brier': float(brier_score_loss(y_test, probs_rf))
    }

    return results, probs_log, probs_rf


def plot_and_save(probs, y_test, out_prefix):
    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.savefig(os.path.join(REPORTS_DIR, f'{out_prefix}_roc.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR')
    plt.savefig(os.path.join(REPORTS_DIR, f'{out_prefix}_pr.png'), bbox_inches='tight')
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_data()
    SMOTE = try_import_smote()
    comparison = {}

    # Baseline (no resampling) - train fresh models for fair comparison
    base_results, base_probs_log, base_probs_rf = train_and_eval(X_train, y_train, X_test, y_test, name_suffix='_base')
    comparison.update(base_results)
    plot_and_save(base_probs_log, y_test, 'base_logreg')
    plot_and_save(base_probs_rf, y_test, 'base_rf')

    if SMOTE is None:
        print('SMOTE not available; skipping resampling experiment. To run SMOTE install imbalanced-learn.')
        with open(os.path.join(REPORTS_DIR, 'resampling_comparison.json'), 'w') as f:
            json.dump({'comparison': comparison, 'note': 'SMOTE not available'}, f, indent=2)
        return

    # Apply SMOTE on training set
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print('After SMOTE, class counts:', np.bincount(y_res))

    res_results, res_probs_log, res_probs_rf = train_and_eval(X_res, y_res, X_test, y_test, name_suffix='_smote')
    comparison.update(res_results)
    plot_and_save(res_probs_log, y_test, 'smote_logreg')
    plot_and_save(res_probs_rf, y_test, 'smote_rf')

    # Save comparison
    out = {'comparison': comparison}
    with open(os.path.join(REPORTS_DIR, 'resampling_comparison.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print('Resampling experiment completed. Results saved to reports/resampling_comparison.json')


if __name__ == '__main__':
    main()
