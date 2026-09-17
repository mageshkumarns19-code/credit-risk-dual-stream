"""
explore_datasets.py
--------------------
Single-file exploratory data analysis (EDA) for both credit-risk datasets:
  1. UCI_Credit_Card.csv  (Taiwan, 30,000 rows - temporal features)
  2. german_credit.csv    (Germany, 1,000 rows - static features)

Run from PowerShell:
    cd "D:\MITS\08-09-26\Dataset"
    python explore_datasets.py

Requires: pandas, numpy, matplotlib, seaborn
Install once with:
    pip install pandas numpy matplotlib seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # saves plots to files instead of popping windows (safer on Windows/PowerShell)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. CONFIG - edit these paths if your files live somewhere else
# ------------------------------------------------------------------
TAIWAN_PATH = "UCI_Credit_Card.csv"
GERMAN_PATH = "german_credit.csv"
OUTPUT_DIR  = "eda_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")


def explore_dataset(name, path, target_col, categorical_cols=None):
    print("\n" + "=" * 70)
    print(f"  EXPLORING: {name}")
    print("=" * 70)

    if not os.path.exists(path):
        print(f"[!] File not found: {path}  -- skipping this dataset.")
        return None

    df = pd.read_csv(path)

    # --- Basic shape & structure -----------------------------------
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn dtypes:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    # --- Missing values ----------------------------------------------
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print("\nMissing values per column:")
    print(nulls if not nulls.empty else "None found.")

    # --- Duplicate rows -----------------------------------------------
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}")

    # --- Target variable distribution ----------------------------------
    if target_col in df.columns:
        print(f"\nTarget column '{target_col}' distribution:")
        print(df[target_col].value_counts())
        print(df[target_col].value_counts(normalize=True).round(4))

        plt.figure(figsize=(4, 4))
        df[target_col].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
        plt.title(f"{name}: Target Class Balance")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_target_balance.png"))
        plt.close()

    # --- Summary statistics for numeric columns -------------------------
    numeric_df = df.select_dtypes(include=[np.number])
    print("\nSummary statistics (numeric columns):")
    print(numeric_df.describe().T)

    # --- Correlation heatmap (numeric columns only) ----------------------
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(12, 10))
        corr = numeric_df.corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        plt.title(f"{name}: Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_correlation_heatmap.png"))
        plt.close()

    # --- Categorical column value counts ---------------------------------
    if categorical_cols:
        print("\nCategorical column distributions:")
        for col in categorical_cols:
            if col in df.columns:
                print(f"\n-- {col} --")
                print(df[col].value_counts())

    # --- Histograms for a handful of numeric columns ----------------------
    cols_to_plot = numeric_df.columns[:6]  # first 6 numeric cols, keep it light
    if len(cols_to_plot) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            axes[i].hist(df[col].dropna(), bins=30, color="#4C72B0")
            axes[i].set_title(col)
        for j in range(len(cols_to_plot), len(axes)):
            fig.delaxes(axes[j])
        plt.suptitle(f"{name}: Feature Distributions")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_histograms.png"))
        plt.close()

    print(f"\n[OK] Plots saved to '{OUTPUT_DIR}/' with prefix '{name}_'")
    return df


if __name__ == "__main__":
    # -----------------------------------------------------------
    # Dataset 1: Taiwan credit card default (temporal features)
    # -----------------------------------------------------------
    taiwan_df = explore_dataset(
        name="Taiwan_Credit",
        path=TAIWAN_PATH,
        target_col="default.payment.next.month",
        categorical_cols=["SEX", "EDUCATION", "MARRIAGE"]
    )

    # -----------------------------------------------------------
    # Dataset 2: German credit (static, categorical-heavy)
    # -----------------------------------------------------------
    german_df = explore_dataset(
        name="German_Credit",
        path=GERMAN_PATH,
        target_col="credit_rating",
        categorical_cols=[
            "account_status", "credit_history", "purpose", "savings",
            "employment", "personal_status", "housing", "job"
        ]
    )

    print("\n" + "=" * 70)
    print("  EDA COMPLETE FOR BOTH DATASETS")
    print(f"  Check the '{OUTPUT_DIR}' folder for saved charts.")
    print("=" * 70)
