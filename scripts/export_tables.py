"""
export_tables.py
-------------------
Exports results_log.json into clean, paper-ready CSV tables.

Outputs (saved to tables_output/):
    table1_taiwan_model_comparison.csv   - all Taiwan models, CV + test metrics
    table2_german_model_comparison.csv   - all German models, CV + test metrics
    table3_ablation_comparison.csv       - Dual-Stream vs Ablation vs Ensemble (Taiwan)
    table4_generalization_comparison.csv - Static-only model, Taiwan vs German

Run from PowerShell:
    cd "D:\MITS\08-09-26\Dataset"
    python export_tables.py
"""

import json
import os
import csv
import pandas as pd

RESULTS_LOG = "results_log.json"
OUTPUT_DIR = "tables_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_KEYS = ["auc", "f1", "accuracy", "precision", "recall"]


def load_latest_by_label():
    with open(RESULTS_LOG, "r") as f:
        results = json.load(f)
    latest = {}
    for entry in results:
        latest[entry["label"]] = entry  # later entries overwrite earlier ones (keeps most recent run)
    return latest


def write_table(filename, rows, header):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"    [Saved] {filename}  ({len(rows)} rows)")


def format_row(short_name, entry):
    row = [short_name]
    for k in METRIC_KEYS:
        row.append(f"{entry['cv_mean'][k]:.4f}")
    for k in METRIC_KEYS:
        row.append(f"{entry['cv_std'][k]:.4f}")
    for k in METRIC_KEYS:
        row.append(f"{entry['test'][k]:.4f}")
    return row


TABLE_HEADER = (
    ["Model"]
    + [f"CV_{k}_mean" for k in METRIC_KEYS]
    + [f"CV_{k}_std" for k in METRIC_KEYS]
    + [f"Test_{k}" for k in METRIC_KEYS]
)


def make_dataset_summary_table():
    """Table 5: basic descriptive stats for both datasets, read directly
    from the raw CSVs so it always reflects the actual data used."""
    rows = []
    if os.path.exists("UCI_Credit_Card.csv"):
        df = pd.read_csv("UCI_Credit_Card.csv")
        n = len(df)
        n_default = int(df["default.payment.next.month"].sum())
        rows.append([
            "Taiwan Credit Card Default", n, df.shape[1] - 2,  # exclude ID + target
            n_default, n - n_default, f"{n_default/n*100:.2f}%",
            "UCI ML Repository (Yeh & Lien, 2009)"
        ])
    if os.path.exists("german_credit.csv"):
        df = pd.read_csv("german_credit.csv")
        n = len(df)
        n_bad = int(df["credit_rating"].sum())
        rows.append([
            "German Credit (Statlog)", n, df.shape[1] - 1,
            n_bad, n - n_bad, f"{n_bad/n*100:.2f}%",
            "UCI ML Repository (Hofmann, 1994)"
        ])
    header = ["Dataset", "N_Samples", "N_Features", "N_Positive_Class",
              "N_Negative_Class", "Positive_Class_Rate", "Source"]
    write_table("table5_dataset_summary.csv", rows, header)


def make_hyperparameter_table():
    """Table 6: model configuration, hardcoded to match the locked values in
    model.py (kept in sync manually -- these are architecture/training
    constants, not learned values, so they don't need to be read from a log)."""
    rows = [
        ["Temporal Stream", "GRU hidden size", "32"],
        ["Temporal Stream", "Attention mechanism", "Additive self-attention over 6 timesteps"],
        ["Static Stream", "Hidden layers", "2 (32 units each, ReLU, dropout=0.2)"],
        ["Fusion", "Mechanism", "Learned sigmoid gate over concatenated [temporal; static] representations"],
        ["Classifier Head", "Hidden layer", "1 (32 units, ReLU, dropout=0.2) -> 1 output logit"],
        ["Training", "Optimizer", "Adam"],
        ["Training", "Learning rate", "1e-3"],
        ["Training", "Batch size", "256"],
        ["Training", "Max epochs", "30"],
        ["Training", "Early stopping patience", "5 epochs (monitored on validation AUC)"],
        ["Training", "Loss function", "Binary Cross-Entropy with Logits"],
        ["Experimental Setup", "Train/Test split", "80% / 20%, stratified"],
        ["Experimental Setup", "Cross-validation", "5-fold Stratified K-Fold (on training set only)"],
        ["Experimental Setup", "Random seed", "42 (fixed across all experiments)"],
        ["Preprocessing", "Feature scaling", "StandardScaler, fit per-fold (train portion only)"],
        ["Preprocessing", "Deduplication", "Exact feature-vector duplicates removed before splitting"],
    ]
    header = ["Component", "Hyperparameter", "Value"]
    write_table("table6_hyperparameters.csv", rows, header)


if __name__ == "__main__":
    if not os.path.exists(RESULTS_LOG):
        print(f"[!] {RESULTS_LOG} not found in this folder.")
        exit(1)

    latest = load_latest_by_label()
    print("Exporting tables...")

    print("\n[Dataset & configuration tables]")
    make_dataset_summary_table()
    make_hyperparameter_table()

    print("\n[Results comparison tables]")

    # ---------------- Table 1: Taiwan model comparison ----------------
    taiwan_models = [
        ("Dual-Stream (Proposed)", "Taiwan - Dual-Stream (Temporal GRU+Attention + Static + Gated Fusion)"),
        ("Ablation (No Temporal)", "Taiwan - Ablation (Flattened features, no temporal structure)"),
        ("Logistic Regression", "Taiwan - Baseline: LogisticRegression"),
        ("Random Forest", "Taiwan - Baseline: RandomForest"),
        ("XGBoost", "Taiwan - Baseline: XGBoost"),
        ("SVM (RBF)", "Taiwan - Baseline: SVM_RBF"),
    ]
    rows = [format_row(name, latest[label]) for name, label in taiwan_models if label in latest]
    write_table("table1_taiwan_model_comparison.csv", rows, TABLE_HEADER)

    # ---------------- Table 2: German model comparison ----------------
    german_models = [
        ("Static-Only (Generalization Check)", "German - Static-Only (Generalization Check)"),
        ("Logistic Regression", "German - Baseline: LogisticRegression"),
        ("Random Forest", "German - Baseline: RandomForest"),
        ("XGBoost", "German - Baseline: XGBoost"),
        ("SVM (RBF)", "German - Baseline: SVM_RBF"),
    ]
    rows = [format_row(name, latest[label]) for name, label in german_models if label in latest]
    write_table("table2_german_model_comparison.csv", rows, TABLE_HEADER)

    # ---------------- Table 3: Ablation + Ensemble comparison ----------------
    ensemble_label = next((l for l in latest if l.startswith("Taiwan - Ensemble")), None)
    ablation_models = [
        ("Dual-Stream (Proposed)", "Taiwan - Dual-Stream (Temporal GRU+Attention + Static + Gated Fusion)"),
        ("Ablation (No Temporal)", "Taiwan - Ablation (Flattened features, no temporal structure)"),
    ]
    rows = [format_row(name, latest[label]) for name, label in ablation_models if label in latest]
    if ensemble_label:
        rows.append(format_row("Ensemble (Dual-Stream + XGBoost)", latest[ensemble_label]))
    write_table("table3_ablation_ensemble_comparison.csv", rows, TABLE_HEADER)

    # ---------------- Table 4: Generalization comparison ----------------
    generalization_models = [
        ("Taiwan (Static-Only)", "Taiwan - Ablation (Flattened features, no temporal structure)"),
        ("German (Static-Only)", "German - Static-Only (Generalization Check)"),
    ]
    rows = [format_row(name, latest[label]) for name, label in generalization_models if label in latest]
    write_table("table4_generalization_comparison.csv", rows, TABLE_HEADER)

    print(f"\nAll tables saved to '{OUTPUT_DIR}/'")
