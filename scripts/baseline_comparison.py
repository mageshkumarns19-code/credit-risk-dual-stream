"""
baseline_comparison.py
------------------------
Runs standard baseline classifiers under the SAME experimental protocol as
the locked dual-stream model (model.py), using the SAME preprocessed data
and the SAME 5-fold stratified CV indices, so results are directly comparable.

Baselines: Logistic Regression, Random Forest, XGBoost, SVM (RBF kernel)
    Note: SVM (RBF) is trained on a stratified subsample (capped at
    MAX_SVM_TRAIN_SAMPLES rows) when the training set is large, since RBF-SVM
    scales poorly (O(n^2)-O(n^3)) with sample size. This only affects SVM's
    training data; test-set evaluation always uses the full test set, and it
    only kicks in for Taiwan (German's ~800-row folds never trigger it). This
    is disclosed in the paper's baseline-setup section for transparency.

For Taiwan: baselines get the FULL flattened feature set (static features +
all 18 raw temporal values across the 6 months) -- not the averaged ablation
version used in model.py's ablation experiment. This gives baselines the
fairest possible shot with all available information, so the dual-stream
model's advantage (if any) reflects genuine architectural benefit, not an
information handicap on the baselines.

For German: same static feature matrix used for the generalization check.

Run from PowerShell (after preprocess_datasets.py has produced the .npz files):
    cd "D:\MITS\08-09-26\Dataset"
    pip install scikit-learn xgboost numpy
    python baseline_comparison.py

Outputs:
    Appends every baseline's CV + test metrics to results_log.json
    (same file model.py writes to), tagged with model_version = "baseline".
"""

import os
import json
import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from model import log_results as _log_results

PROCESSED_DIR = "processed"
RESULTS_LOG = "results_log.json"
RANDOM_STATE = 42
MAX_SVM_TRAIN_SAMPLES = 4000  # RBF-SVM scales poorly (O(n^2)-O(n^3)); cap training rows for speed

# NOTE: preprocess_datasets.py now saves UNSCALED features (fixed in v1.1 to
# avoid statistical leakage -- see model.py). Each baseline below is wrapped
# in a Pipeline(StandardScaler(), model), so scaling is fit fresh on
# whatever data .fit() is called with (a CV fold's train rows, or the full
# train set for the final fit) -- never on validation/test rows. Tree models
# (RF, XGBoost) don't need scaling but it doesn't harm them either, so using
# the same pipeline everywhere keeps this script simple and consistent.


class CappedSVC(BaseEstimator, ClassifierMixin):
    """RBF-SVM wrapper that subsamples the training set (stratified) when it
    exceeds MAX_SVM_TRAIN_SAMPLES, so CV/refit stays fast on large datasets
    like Taiwan (~19-24k rows). Test-set prediction is never subsampled."""
    def __init__(self, max_train_samples=MAX_SVM_TRAIN_SAMPLES, random_state=RANDOM_STATE):
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.model_ = SVC(kernel="rbf", probability=True, random_state=random_state)

    def fit(self, X, y):
        if len(y) > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = min(len(pos_idx), int(self.max_train_samples * (len(pos_idx) / len(y))))
            n_neg = self.max_train_samples - n_pos
            sample_idx = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
            ])
            X, y = X[sample_idx], y[sample_idx]
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

BASELINE_MODELS = {
    "LogisticRegression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "RandomForest": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "XGBoost": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ]),
    "SVM_RBF": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CappedSVC()),
    ]),
}


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def log_results(label, fold_metrics_list, test_metrics):
    # Reuses model.py's log_results (single source of truth for the log
    # format, including the cv_folds field needed for significance testing),
    # just tagged with "baseline" instead of the Dual-Stream model version.
    _log_results(label, fold_metrics_list, test_metrics, model_version="baseline")


def run_baseline(name, build_model_fn, X_train, y_train, fold_train_idx, fold_val_idx, X_test, y_test, dataset_label):
    """Single reusable runner: 5-fold CV -> summarize -> refit on full train -> evaluate on test.
    Used for every (dataset, baseline model) combination, so this logic is written once."""
    label = f"{dataset_label} - Baseline: {name}"
    print(f"\n-- {label} --")

    fold_metrics_list = []
    for fold_i, (tr_idx, val_idx) in enumerate(zip(fold_train_idx, fold_val_idx), start=1):
        tr_idx, val_idx = np.array(tr_idx, dtype=int), np.array(val_idx, dtype=int)
        model = build_model_fn()
        model.fit(X_train[tr_idx], y_train[tr_idx])
        y_prob = model.predict_proba(X_train[val_idx])[:, 1]
        fold_metrics = compute_metrics(y_train[val_idx], y_prob)
        fold_metrics_list.append(fold_metrics)
        print(f"    Fold {fold_i}: auc={fold_metrics['auc']:.4f} f1={fold_metrics['f1']:.4f}")

    cv_auc_mean = np.mean([m["auc"] for m in fold_metrics_list])
    cv_auc_std = np.std([m["auc"] for m in fold_metrics_list])
    print(f"    CV AUC: {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")

    # Refit on full training set, evaluate once on held-out test
    final_model = build_model_fn()
    final_model.fit(X_train, y_train)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_prob)
    print(f"    TEST: auc={test_metrics['auc']:.4f} f1={test_metrics['f1']:.4f} "
          f"acc={test_metrics['accuracy']:.4f} prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f}")

    log_results(label, fold_metrics_list, test_metrics)
    return fold_metrics_list, test_metrics


def build_taiwan_flat_features(d):
    """Full flattened feature set for Taiwan: static features + all 18 raw
    temporal values (not averaged), giving baselines maximum information."""
    def flatten(X_static, X_seq):
        n = X_seq.shape[0]
        X_seq_flat = X_seq.reshape(n, -1)  # (N, 6*3=18)
        return np.hstack([X_static, X_seq_flat])

    X_train = flatten(d["X_static_train"], d["X_seq_train"])
    X_test = flatten(d["X_static_test"], d["X_seq_test"])
    return X_train, X_test


if __name__ == "__main__":
    all_results = {}

    # ---------------- Taiwan baselines ----------------
    taiwan_path = os.path.join(PROCESSED_DIR, "taiwan_processed.npz")
    if os.path.exists(taiwan_path):
        d = np.load(taiwan_path, allow_pickle=True)
        X_train, X_test = build_taiwan_flat_features(d)
        y_train, y_test = d["y_train"], d["y_test"]
        fold_train_idx, fold_val_idx = d["fold_train_idx"], d["fold_val_idx"]

        print("\n" + "=" * 70)
        print("  TAIWAN BASELINES (full flattened features, 26 total)")
        print("=" * 70)
        for name, build_fn in BASELINE_MODELS.items():
            cv_metrics, test_metrics = run_baseline(
                name, build_fn, X_train, y_train, fold_train_idx, fold_val_idx,
                X_test, y_test, dataset_label="Taiwan"
            )
            all_results[f"Taiwan_{name}"] = {"cv": cv_metrics, "test": test_metrics}
    else:
        print(f"[!] {taiwan_path} not found. Run preprocess_datasets.py first.")

    # ---------------- German baselines ----------------
    german_path = os.path.join(PROCESSED_DIR, "german_processed.npz")
    if os.path.exists(german_path):
        d = np.load(german_path, allow_pickle=True)
        X_train, X_test = d["X_train"], d["X_test"]
        y_train, y_test = d["y_train"], d["y_test"]
        fold_train_idx, fold_val_idx = d["fold_train_idx"], d["fold_val_idx"]

        print("\n" + "=" * 70)
        print("  GERMAN BASELINES")
        print("=" * 70)
        for name, build_fn in BASELINE_MODELS.items():
            cv_metrics, test_metrics = run_baseline(
                name, build_fn, X_train, y_train, fold_train_idx, fold_val_idx,
                X_test, y_test, dataset_label="German"
            )
            all_results[f"German_{name}"] = {"cv": cv_metrics, "test": test_metrics}
    else:
        print(f"[!] {german_path} not found. Run preprocess_datasets.py first.")

    # ---------------- Summary table ----------------
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON SUMMARY (Test set)")
    print("=" * 70)
    print(f"{'Model':35s} {'AUC':>8s} {'F1':>8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>8s}")
    for key, res in all_results.items():
        t = res["test"]
        print(f"{key:35s} {t['auc']:8.4f} {t['f1']:8.4f} {t['accuracy']:10.4f} {t['precision']:10.4f} {t['recall']:8.4f}")

    print("\n" + "=" * 70)
    print("  ALL BASELINES COMPLETE")
    print(f"  Compare against your dual-stream model's results already in {RESULTS_LOG}")
    print("=" * 70)
