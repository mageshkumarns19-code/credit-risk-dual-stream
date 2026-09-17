"""
preprocess_datasets.py
------------------------
Preprocesses both credit-risk datasets and builds the experimental setup:
    Train (80%) -> 5-Fold Stratified Cross-Validation
    Test  (20%) -> Held out, untouched until final evaluation

Taiwan dataset  -> dual-stream ready: static features + (6-month) temporal sequence
German dataset  -> static-only, used as a generalization check

Run from PowerShell:
    cd "D:\MITS\08-09-26\Dataset"
    python preprocess_datasets.py

Requires: pandas, numpy, scikit-learn
    pip install pandas numpy scikit-learn
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
TAIWAN_PATH = "UCI_Credit_Card.csv"
GERMAN_PATH = "german_credit.csv"
OUTPUT_DIR  = "processed"
RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================================
# PART 1: TAIWAN DATASET  (temporal + static dual-stream)
# ==================================================================
def preprocess_taiwan(path):
    print("\n" + "=" * 70)
    print("  PREPROCESSING: Taiwan Credit Card Dataset")
    print("=" * 70)

    df = pd.read_csv(path)
    df = df.drop(columns=["ID"])

    # --- Remove duplicate FEATURE rows FIRST (before split) ---------------
    # Dedup on features only (excluding target) -- catches cases where two
    # clients have identical feature vectors but different recorded targets,
    # which a full-row dedup would miss and which would otherwise let the
    # same feature vector appear in both train and test after splitting.
    feature_cols_for_dedup = [c for c in df.columns if c != "default.payment.next.month"]
    n_before = len(df)
    df = df.drop_duplicates(subset=feature_cols_for_dedup, keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"[Dedup] Removed {n_removed} rows with duplicate feature vectors ({n_before} -> {len(df)})")

    # --- Clean undocumented category codes -------------------------
    # EDUCATION: valid = 1,2,3,4 ; codes 0,5,6 are undocumented -> merge into "4" (others)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # MARRIAGE: valid = 1,2,3 ; code 0 is undocumented -> merge into "3" (others)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    print(f"[Clean] EDUCATION unique values now: {sorted(df['EDUCATION'].unique())}")
    print(f"[Clean] MARRIAGE unique values now:  {sorted(df['MARRIAGE'].unique())}")

    # --- Feature engineering (static, non-temporal) -------------------
    # Overall utilization ratio: average bill relative to credit limit
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]

    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
    df["AVG_PAY_AMT"] = df[pay_amt_cols].mean(axis=1)
    df["UTILIZATION_RATIO"] = df["AVG_BILL_AMT"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["UTILIZATION_RATIO"] = df["UTILIZATION_RATIO"].fillna(0)
    df["PAYMENT_RATIO"] = df["AVG_PAY_AMT"] / df["AVG_BILL_AMT"].replace(0, np.nan)
    df["PAYMENT_RATIO"] = df["PAYMENT_RATIO"].replace([np.inf, -np.inf], 0).fillna(0)
    df["MAX_DELINQUENCY"] = df[pay_status_cols].max(axis=1)  # worst repayment status across 6 months

    # --- Define feature groups ---------------------------------------
    static_cols = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "UTILIZATION_RATIO", "PAYMENT_RATIO", "MAX_DELINQUENCY"
    ]
    target_col = "default.payment.next.month"

    # Temporal sequence: for each of the 6 months, group [PAY_status, BILL_AMT, PAY_AMT]
    # PAY_0 corresponds to the most recent month, so order is month1..month6 (oldest->newest not guaranteed,
    # but consistent ordering is what matters for the sequence model)
    month_pay_status = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    month_bill = bill_cols
    month_payamt = pay_amt_cols

    X_static = df[static_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    # Build sequence array: shape (N, 6 timesteps, 3 features per timestep)
    seq_list = []
    for i in range(6):
        month_features = np.stack([
            df[month_pay_status[i]].values,
            df[month_bill[i]].values,
            df[month_payamt[i]].values
        ], axis=1)  # shape (N, 3)
        seq_list.append(month_features)
    X_seq = np.stack(seq_list, axis=1).astype(np.float32)  # shape (N, 6, 3)

    print(f"[Shapes] X_static: {X_static.shape} | X_seq: {X_seq.shape} | y: {y.shape}")

    # --- Train / Test split (stratified) -------------------------------
    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    Xs_train, Xs_test = X_static[idx_train], X_static[idx_test]
    Xt_train, Xt_test = X_seq[idx_train], X_seq[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"[Split] Train: {len(y_train)}  ({y_train.mean():.4f} positive rate)")
    print(f"[Split] Test:  {len(y_test)}  ({y_test.mean():.4f} positive rate)")

    # NOTE: Features are saved UNSCALED here on purpose. Scaling is fit
    # per-CV-fold (and separately for the final train/test fit) inside
    # model.py's training loop, to avoid any statistical leakage from
    # validation-fold rows influencing the scaler used to transform them.

    # --- 5-Fold Stratified CV indices on the TRAIN set only ---------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(Xs_train, y_train))
    print(f"[CV] Generated {N_FOLDS}-fold stratified split indices on training set.")
    for i, (tr_idx, val_idx) in enumerate(fold_indices):
        print(f"     Fold {i+1}: train={len(tr_idx)}  val={len(val_idx)}  "
              f"val_pos_rate={y_train[val_idx].mean():.4f}")

    # --- Save everything (RAW, unscaled) -----------------------------------
    out_path = os.path.join(OUTPUT_DIR, "taiwan_processed.npz")
    np.savez(
        out_path,
        X_static_train=Xs_train, X_static_test=Xs_test,
        X_seq_train=Xt_train, X_seq_test=Xt_test,
        y_train=y_train, y_test=y_test,
        fold_train_idx=np.array([f[0] for f in fold_indices], dtype=object),
        fold_val_idx=np.array([f[1] for f in fold_indices], dtype=object),
        static_cols=np.array(static_cols),
    )
    print(f"[Saved] {out_path}  (unscaled -- scaling is done per-fold in model.py)")
    return out_path


# ==================================================================
# PART 2: GERMAN CREDIT DATASET  (static-only generalization check)
# ==================================================================
def preprocess_german(path):
    print("\n" + "=" * 70)
    print("  PREPROCESSING: German Credit Dataset")
    print("=" * 70)

    df = pd.read_csv(path)
    target_col = "credit_rating"

    categorical_cols = [
        "account_status", "credit_history", "purpose", "savings",
        "employment", "personal_status", "guarantors", "property",
        "other_installments", "housing", "job", "phone", "foreign_worker"
    ]
    numeric_cols = [
        "months", "credit_amount", "installment_rate", "residence",
        "age", "credit_cards", "dependents"
    ]

    X_df = df[categorical_cols + numeric_cols].copy()
    y = df[target_col].values.astype(np.int64)

    # --- Train / Test split (stratified) FIRST, to avoid leakage in encoding ---
    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train_df, X_test_df = X_df.iloc[idx_train].reset_index(drop=True), X_df.iloc[idx_test].reset_index(drop=True)
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"[Split] Train: {len(y_train)}  ({y_train.mean():.4f} positive rate)")
    print(f"[Split] Test:  {len(y_test)}  ({y_test.mean():.4f} positive rate)")

    # --- One-hot encode categoricals (FIT ON TRAIN ONLY) ------------------
    # Category discovery is structural, not a leakage-sensitive statistic,
    # so fitting once on the full training set here is fine.
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xcat_train = ohe.fit_transform(X_train_df[categorical_cols])
    Xcat_test = ohe.transform(X_test_df[categorical_cols])
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)

    # NOTE: Numeric columns are kept UNSCALED here. Scaling is fit per-fold
    # (and separately for the final train/test fit) inside model.py, for the
    # same leakage-avoidance reason as the Taiwan preprocessing above.
    Xnum_train = X_train_df[numeric_cols].values.astype(np.float32)
    Xnum_test = X_test_df[numeric_cols].values.astype(np.float32)

    X_train = np.hstack([Xnum_train, Xcat_train]).astype(np.float32)
    X_test = np.hstack([Xnum_test, Xcat_test]).astype(np.float32)
    all_feature_names = numeric_cols + list(cat_feature_names)
    n_numeric = len(numeric_cols)  # model.py needs this to know which columns to scale

    print(f"[Shapes] X_train: {X_train.shape} | X_test: {X_test.shape}")

    # --- 5-Fold Stratified CV indices on TRAIN set only ----------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(X_train, y_train))
    print(f"[CV] Generated {N_FOLDS}-fold stratified split indices on training set.")
    for i, (tr_idx, val_idx) in enumerate(fold_indices):
        print(f"     Fold {i+1}: train={len(tr_idx)}  val={len(val_idx)}  "
              f"val_pos_rate={y_train[val_idx].mean():.4f}")

    # --- Save everything ---------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, "german_processed.npz")
    np.savez(
        out_path,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        fold_train_idx=np.array([f[0] for f in fold_indices], dtype=object),
        fold_val_idx=np.array([f[1] for f in fold_indices], dtype=object),
        feature_names=np.array(all_feature_names),
        n_numeric=n_numeric,
    )
    print(f"[Saved] {out_path}  (unscaled -- scaling is done per-fold in model.py)")
    return out_path


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    if os.path.exists(TAIWAN_PATH):
        preprocess_taiwan(TAIWAN_PATH)
    else:
        print(f"[!] {TAIWAN_PATH} not found, skipping.")

    if os.path.exists(GERMAN_PATH):
        preprocess_german(GERMAN_PATH)
    else:
        print(f"[!] {GERMAN_PATH} not found, skipping.")

    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE")
    print(f"  Processed .npz files saved in '{OUTPUT_DIR}/'")
    print("  Load them in your training script with:")
    print("      data = np.load('processed/taiwan_processed.npz', allow_pickle=True)")
    print("=" * 70)
