"""
ensemble_check.py
--------------------
Checks whether ensembling the locked Dual-Stream model with XGBoost (the
strongest baseline) beats either model alone on Taiwan.

Method (leakage-safe):
    1. For each of the 5 CV folds: train Dual-Stream AND XGBoost on the
       fold's training rows, predict on the fold's validation rows.
       Collecting these across all 5 folds gives Out-Of-Fold (OOF)
       predictions covering the entire training set -- each prediction
       comes from a model that never saw that row during training.
    2. Search blend weights w*DualStream + (1-w)*XGBoost against the OOF
       predictions (still no test-set contact) to find the best blend.
    3. Refit both models on the FULL training set, generate test
       predictions, apply the chosen blend weight, evaluate ONCE on the
       held-out test set.

Reuses functions directly from model.py and baseline_comparison.py --
no logic is duplicated.

Run from PowerShell (same folder as model.py, baseline_comparison.py,
and processed/taiwan_processed.npz):
    cd "D:\MITS\08-09-26\Dataset"
    python ensemble_check.py
"""

import os
import numpy as np
import torch
import torch.nn as nn

from model import (
    DualStreamModel, make_loader, make_scalers, apply_scalers,
    train_with_early_stopping, run_epoch, compute_metrics, log_results,
    DEVICE, BATCH_SIZE, RANDOM_STATE
)
from baseline_comparison import BASELINE_MODELS, build_taiwan_flat_features

PROCESSED_DIR = "processed"
WEIGHT_GRID = np.arange(0.0, 1.01, 0.05)  # search DualStream weight from 0 to 1


def get_dualstream_oof_and_test_probs(d, fold_train_idx, fold_val_idx, n_static, n_seq_feat):
    """Trains the Dual-Stream model per-fold, returns OOF probs on train +
    final-fit probs on test. Reuses model.py's training/scaling utilities."""
    y_train = d["y_train"]
    oof_probs = np.zeros(len(y_train))

    for fold_i, (tr_idx, val_idx) in enumerate(zip(fold_train_idx, fold_val_idx), start=1):
        tr_idx, val_idx = np.array(tr_idx, dtype=int), np.array(val_idx, dtype=int)
        print(f"    [DualStream] Fold {fold_i}/{len(fold_train_idx)}")

        raw_tr = [d["X_static_train"][tr_idx], d["X_seq_train"][tr_idx]]
        raw_val = [d["X_static_train"][val_idx], d["X_seq_train"][val_idx]]
        scalers = make_scalers(raw_tr, [n_static, n_seq_feat])
        scaled_tr = apply_scalers(raw_tr, scalers)
        scaled_val = apply_scalers(raw_val, scalers)

        train_tensors = [torch.tensor(x, dtype=torch.float32) for x in scaled_tr]
        train_tensors.append(torch.tensor(y_train[tr_idx], dtype=torch.float32))
        val_tensors = [torch.tensor(x, dtype=torch.float32) for x in scaled_val]
        val_tensors.append(torch.tensor(y_train[val_idx], dtype=torch.float32))

        train_loader = make_loader(train_tensors, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_tensors, BATCH_SIZE, shuffle=False)

        model = DualStreamModel(n_static, n_seq_feat)
        model, _ = train_with_early_stopping(model, train_loader, val_loader, dual_stream=True)
        _, val_probs = run_epoch(model, val_loader, None, nn.BCEWithLogitsLoss(), train=False, dual_stream=True)
        oof_probs[val_idx] = val_probs

    # Final fit on full train, predict on test
    print("    [DualStream] Final fit on full train set...")
    final_scalers = make_scalers([d["X_static_train"], d["X_seq_train"]], [n_static, n_seq_feat])
    scaled_full_train = apply_scalers([d["X_static_train"], d["X_seq_train"]], final_scalers)
    scaled_test = apply_scalers([d["X_static_test"], d["X_seq_test"]], final_scalers)

    n = len(y_train)
    val_cut = int(n * 0.1)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    final_val_idx, final_train_idx = perm[:val_cut], perm[val_cut:]

    final_train_tensors = [torch.tensor(x[final_train_idx], dtype=torch.float32) for x in scaled_full_train]
    final_train_tensors.append(torch.tensor(y_train[final_train_idx], dtype=torch.float32))
    final_val_tensors = [torch.tensor(x[final_val_idx], dtype=torch.float32) for x in scaled_full_train]
    final_val_tensors.append(torch.tensor(y_train[final_val_idx], dtype=torch.float32))

    final_train_loader = make_loader(final_train_tensors, BATCH_SIZE, shuffle=True)
    final_val_loader = make_loader(final_val_tensors, BATCH_SIZE, shuffle=False)
    final_model = DualStreamModel(n_static, n_seq_feat)
    final_model, _ = train_with_early_stopping(final_model, final_train_loader, final_val_loader, dual_stream=True)

    test_tensors = [torch.tensor(x, dtype=torch.float32) for x in scaled_test]
    test_tensors.append(torch.tensor(d["y_test"], dtype=torch.float32))
    test_loader = make_loader(test_tensors, BATCH_SIZE, shuffle=False)
    _, test_probs = run_epoch(final_model, test_loader, None, nn.BCEWithLogitsLoss(), train=False, dual_stream=True)

    return oof_probs, test_probs


def get_xgb_oof_and_test_probs(X_train, y_train, X_test, fold_train_idx, fold_val_idx):
    """Trains XGBoost (via the Pipeline in baseline_comparison.py) per-fold,
    returns OOF probs on train + final-fit probs on test."""
    oof_probs = np.zeros(len(y_train))
    build_fn = BASELINE_MODELS["XGBoost"]

    for fold_i, (tr_idx, val_idx) in enumerate(zip(fold_train_idx, fold_val_idx), start=1):
        tr_idx, val_idx = np.array(tr_idx, dtype=int), np.array(val_idx, dtype=int)
        print(f"    [XGBoost] Fold {fold_i}/{len(fold_train_idx)}")
        model = build_fn()
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_probs[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]

    print("    [XGBoost] Final fit on full train set...")
    final_model = build_fn()
    final_model.fit(X_train, y_train)
    test_probs = final_model.predict_proba(X_test)[:, 1]
    return oof_probs, test_probs


def find_best_blend_weight(oof_probs_a, oof_probs_b, y_true):
    """Grid-searches the blend weight against OOF predictions only (no test
    contact), returns the weight and its OOF AUC."""
    from sklearn.metrics import roc_auc_score
    best_w, best_auc = 0.5, -1
    for w in WEIGHT_GRID:
        blend = w * oof_probs_a + (1 - w) * oof_probs_b
        auc = roc_auc_score(y_true, blend)
        if auc > best_auc:
            best_auc, best_w = auc, w
    return best_w, best_auc


if __name__ == "__main__":
    taiwan_path = os.path.join(PROCESSED_DIR, "taiwan_processed.npz")
    if not os.path.exists(taiwan_path):
        print(f"[!] {taiwan_path} not found. Run preprocess_datasets.py first.")
        exit(1)

    d = np.load(taiwan_path, allow_pickle=True)
    n_static = d["X_static_train"].shape[1]
    n_seq_feat = d["X_seq_train"].shape[2]
    fold_train_idx, fold_val_idx = d["fold_train_idx"], d["fold_val_idx"]
    y_train, y_test = d["y_train"], d["y_test"]

    X_train_flat, X_test_flat = build_taiwan_flat_features(d)

    print("=" * 70)
    print("  ENSEMBLE CHECK: Dual-Stream + XGBoost")
    print("=" * 70)

    print("\n[Step 1/3] Collecting Dual-Stream OOF + test predictions...")
    ds_oof, ds_test = get_dualstream_oof_and_test_probs(d, fold_train_idx, fold_val_idx, n_static, n_seq_feat)

    print("\n[Step 2/3] Collecting XGBoost OOF + test predictions...")
    xgb_oof, xgb_test = get_xgb_oof_and_test_probs(X_train_flat, y_train, X_test_flat, fold_train_idx, fold_val_idx)

    print("\n[Step 3/3] Searching blend weight on OOF predictions (no test contact)...")
    best_w, best_oof_auc = find_best_blend_weight(ds_oof, xgb_oof, y_train)
    print(f"    Best weight: {best_w:.2f} * DualStream + {1-best_w:.2f} * XGBoost")
    print(f"    OOF AUC at best weight: {best_oof_auc:.4f}")

    # Individual OOF AUCs for reference
    from sklearn.metrics import roc_auc_score
    print(f"    OOF AUC -- DualStream alone: {roc_auc_score(y_train, ds_oof):.4f}")
    print(f"    OOF AUC -- XGBoost alone:    {roc_auc_score(y_train, xgb_oof):.4f}")

    # ---------- Final test-set evaluation of the blend ----------
    blend_test_probs = best_w * ds_test + (1 - best_w) * xgb_test
    ensemble_test_metrics = compute_metrics(y_test, blend_test_probs)
    ds_test_metrics = compute_metrics(y_test, ds_test)
    xgb_test_metrics = compute_metrics(y_test, xgb_test)

    print("\n" + "=" * 70)
    print("  TEST SET COMPARISON")
    print("=" * 70)
    print(f"{'Model':30s} {'AUC':>8s} {'F1':>8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>8s}")
    for name, m in [("DualStream alone", ds_test_metrics),
                    ("XGBoost alone", xgb_test_metrics),
                    (f"Ensemble (w={best_w:.2f})", ensemble_test_metrics)]:
        print(f"{name:30s} {m['auc']:8.4f} {m['f1']:8.4f} {m['accuracy']:10.4f} {m['precision']:10.4f} {m['recall']:8.4f}")

    # ---------- Log to results_log.json (reusing model.py's logger) ----------
    # For the "CV" field here we report the OOF-based blend metrics per fold
    # weight search, approximated as a single-fold-equivalent summary since
    # blending is evaluated globally on OOF, not per-fold.
    oof_metrics_as_single_fold = [compute_metrics(y_train, best_w * ds_oof + (1 - best_w) * xgb_oof)]
    log_results(
        f"Taiwan - Ensemble (DualStream w={best_w:.2f} + XGBoost w={1-best_w:.2f})",
        oof_metrics_as_single_fold,
        ensemble_test_metrics
    )

    print("\n" + "=" * 70)
    improvement = ensemble_test_metrics["auc"] - max(ds_test_metrics["auc"], xgb_test_metrics["auc"])
    if improvement > 0.002:
        print(f"  RESULT: Ensemble improves test AUC by {improvement:+.4f} over the best single model.")
    else:
        print(f"  RESULT: Ensemble does NOT meaningfully improve over the best single model "
              f"(AUC change: {improvement:+.4f}). The Dual-Stream model alone is the simpler,")
        print(f"  equally-good choice -- no need to report an ensemble in the paper.")
    print("=" * 70)
