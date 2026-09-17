"""
generate_graphs.py
---------------------
Generates all 17 figures for the study, covering:

  EDA (4):
    01_taiwan_class_distribution.png
    02_german_class_distribution.png
    03_taiwan_correlation_heatmap.png
    04_taiwan_age_by_default.png

  Model performance comparison (6):
    05_test_auc_comparison_all_models.png
    06_test_f1_comparison_all_models.png
    07_roc_curves_comparison.png
    08_precision_recall_curves_comparison.png
    09_confusion_matrix_dualstream.png
    10_confusion_matrix_xgboost.png

  Training diagnostics (2):
    11_dualstream_training_curve.png
    12_cv_auc_comparison_with_error_bars.png

  Explainability (4):
    13_attention_weight_by_month.png
    14_attention_by_outcome.png
    15_gate_trust_distribution.png
    16_shap_feature_importance.png

  Generalization (1):
    17_dataset_generalization_comparison.png

Reuses functions from model.py and baseline_comparison.py directly --
no logic is duplicated. Retrains fast sklearn baselines (LR, RF, XGBoost)
on the full training set to get test-set probabilities for the ROC/PR
curves (their CV/test metrics are already in results_log.json, but the
raw probability arrays needed for curve-plotting weren't saved).

Run from PowerShell (same folder as everything else built so far):
    cd "D:\MITS\08-09-26\Dataset"
    pip install shap matplotlib seaborn scikit-learn xgboost
    python generate_graphs.py

Outputs saved to graphs_output/
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc

from model import DualStreamModel, make_scalers, apply_scalers, make_loader, run_epoch, RANDOM_STATE
from baseline_comparison import BASELINE_MODELS, build_taiwan_flat_features

PROCESSED_DIR = "processed"
CHECKPOINT_PATH = os.path.join(
    "checkpoints",
    "taiwan__dualstream_temporal_gruplusattention_plus_static_plus_gated_fusion_final_v1.1.pt"
)
GERMAN_CHECKPOINT_PATH = os.path.join(
    "checkpoints", "german__staticonly_generalization_check_final_v1.1.pt"
)
RESULTS_LOG = "results_log.json"
OUTPUT_DIR = "graphs_output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MONTH_LABELS = ["t-5", "t-4", "t-3", "t-2", "t-1", "t (most recent)"]
STATIC_FEATURE_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                         "UTILIZATION_RATIO", "PAYMENT_RATIO", "MAX_DELINQUENCY"]
SEQ_CHANNEL_NAMES = ["PAY_STATUS", "BILL_AMT", "PAY_AMT"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [Saved] {name}")


# ==================================================================
# Load everything once
# ==================================================================
def load_all():
    d_taiwan = np.load(os.path.join(PROCESSED_DIR, "taiwan_processed.npz"), allow_pickle=True)
    d_german = np.load(os.path.join(PROCESSED_DIR, "german_processed.npz"), allow_pickle=True)

    with open(RESULTS_LOG, "r") as f:
        results = json.load(f)

    n_static = d_taiwan["X_static_train"].shape[1]
    n_seq_feat = d_taiwan["X_seq_train"].shape[2]

    scalers = make_scalers([d_taiwan["X_static_train"], d_taiwan["X_seq_train"]], [n_static, n_seq_feat])
    Xs_test, Xt_test = apply_scalers([d_taiwan["X_static_test"], d_taiwan["X_seq_test"]], scalers)
    y_test = d_taiwan["y_test"]

    dual_model = DualStreamModel(n_static, n_seq_feat)
    dual_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    dual_model.to(DEVICE)
    dual_model.eval()

    return {
        "d_taiwan": d_taiwan, "d_german": d_german, "results": results,
        "n_static": n_static, "n_seq_feat": n_seq_feat,
        "Xs_test": Xs_test, "Xt_test": Xt_test, "y_test": y_test,
        "dual_model": dual_model,
    }


def get_dualstream_test_probs(ctx):
    with torch.no_grad():
        xs = torch.tensor(ctx["Xs_test"], dtype=torch.float32).to(DEVICE)
        xt = torch.tensor(ctx["Xt_test"], dtype=torch.float32).to(DEVICE)
        logits, attn, gates = ctx["dual_model"](xs, xt)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs, attn.cpu().numpy(), gates.cpu().numpy()


def get_baseline_test_probs(ctx, name):
    """Quickly refits a baseline (already validated in baseline_comparison.py)
    on the full training set to get test-set probabilities for curve plots."""
    d = ctx["d_taiwan"]
    X_train, X_test = build_taiwan_flat_features(d)
    model = BASELINE_MODELS[name]()
    model.fit(X_train, d["y_train"])
    return model.predict_proba(X_test)[:, 1]


# ==================================================================
# 1-4: EDA
# ==================================================================
def make_eda_graphs(ctx):
    print("\n[EDA graphs]")
    df_taiwan = pd.read_csv("UCI_Credit_Card.csv")
    df_german = pd.read_csv("german_credit.csv")

    # 01: Taiwan class distribution
    plt.figure(figsize=(5, 4))
    counts = df_taiwan["default.payment.next.month"].value_counts().sort_index()
    plt.bar(["No Default", "Default"], counts.values, color=["#55A868", "#C44E52"])
    plt.title("Taiwan Dataset: Class Distribution")
    plt.ylabel("Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 200, str(v), ha="center")
    savefig("01_taiwan_class_distribution.png")

    # 02: German class distribution
    plt.figure(figsize=(5, 4))
    counts = df_german["credit_rating"].value_counts().sort_index()
    plt.bar(["Good Credit", "Bad Credit"], counts.values, color=["#55A868", "#C44E52"])
    plt.title("German Dataset: Class Distribution")
    plt.ylabel("Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 10, str(v), ha="center")
    savefig("02_german_class_distribution.png")

    # 03: Taiwan correlation heatmap
    plt.figure(figsize=(11, 9))
    numeric_df = df_taiwan.drop(columns=["ID"]).select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Taiwan Dataset: Feature Correlation Heatmap")
    savefig("03_taiwan_correlation_heatmap.png")

    # 04: Age distribution by default status
    plt.figure(figsize=(7, 5))
    sns.kdeplot(df_taiwan[df_taiwan["default.payment.next.month"] == 0]["AGE"], label="No Default", fill=True, alpha=0.4)
    sns.kdeplot(df_taiwan[df_taiwan["default.payment.next.month"] == 1]["AGE"], label="Default", fill=True, alpha=0.4)
    plt.title("Age Distribution by Default Status (Taiwan)")
    plt.xlabel("Age")
    plt.legend()
    savefig("04_taiwan_age_by_default.png")


# ==================================================================
# 5-10: Model performance comparison
# ==================================================================
def make_performance_graphs(ctx):
    print("\n[Performance comparison graphs]")
    results = ctx["results"]

    # Pick the latest entry per label (in case of re-runs) for Taiwan-related models
    latest_by_label = {}
    for entry in results:
        latest_by_label[entry["label"]] = entry  # later entries overwrite earlier ones

    taiwan_labels_order = [
        "Taiwan - Dual-Stream (Temporal GRU+Attention + Static + Gated Fusion)",
        "Taiwan - Ablation (Flattened features, no temporal structure)",
        "Taiwan - Baseline: LogisticRegression",
        "Taiwan - Baseline: RandomForest",
        "Taiwan - Baseline: XGBoost",
        "Taiwan - Baseline: SVM_RBF",
    ]
    short_names = ["Dual-Stream", "Ablation", "LogReg", "RandomForest", "XGBoost", "SVM"]

    test_aucs = [latest_by_label[l]["test"]["auc"] for l in taiwan_labels_order if l in latest_by_label]
    test_f1s = [latest_by_label[l]["test"]["f1"] for l in taiwan_labels_order if l in latest_by_label]
    valid_names = [n for l, n in zip(taiwan_labels_order, short_names) if l in latest_by_label]

    # 05: Test AUC comparison
    plt.figure(figsize=(9, 5))
    colors = ["#C44E52" if n == "Dual-Stream" else "#4C72B0" for n in valid_names]
    plt.bar(valid_names, test_aucs, color=colors)
    plt.ylabel("Test AUC")
    plt.title("Test AUC Comparison Across Models (Taiwan)")
    plt.ylim(0.6, 0.85)
    for i, v in enumerate(test_aucs):
        plt.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    savefig("05_test_auc_comparison_all_models.png")

    # 06: Test F1 comparison
    plt.figure(figsize=(9, 5))
    plt.bar(valid_names, test_f1s, color=colors)
    plt.ylabel("Test F1")
    plt.title("Test F1 Comparison Across Models (Taiwan)")
    for i, v in enumerate(test_f1s):
        plt.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    savefig("06_test_f1_comparison_all_models.png")

    # Get probabilities for ROC/PR curves
    y_test = ctx["y_test"]
    ds_probs, attn, gates = get_dualstream_test_probs(ctx)
    print("    Refitting fast baselines (LR, RF, XGBoost) for curve plotting...")
    lr_probs = get_baseline_test_probs(ctx, "LogisticRegression")
    rf_probs = get_baseline_test_probs(ctx, "RandomForest")
    xgb_probs = get_baseline_test_probs(ctx, "XGBoost")

    curve_models = [
        ("Dual-Stream", ds_probs, "#C44E52"),
        ("XGBoost", xgb_probs, "#4C72B0"),
        ("Random Forest", rf_probs, "#55A868"),
        ("Logistic Regression", lr_probs, "#8172B2"),
    ]

    # 07: ROC curves
    plt.figure(figsize=(7, 6))
    for name, probs, color in curve_models:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=color)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison (Taiwan Test Set)")
    plt.legend()
    savefig("07_roc_curves_comparison.png")

    # 08: Precision-Recall curves
    plt.figure(figsize=(7, 6))
    for name, probs, color in curve_models:
        prec, rec, _ = precision_recall_curve(y_test, probs)
        plt.plot(rec, prec, label=name, color=color)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves Comparison (Taiwan Test Set)")
    plt.legend()
    savefig("08_precision_recall_curves_comparison.png")

    # 09-10: Confusion matrices
    for fname, title, probs in [
        ("09_confusion_matrix_dualstream.png", "Confusion Matrix: Dual-Stream", ds_probs),
        ("10_confusion_matrix_xgboost.png", "Confusion Matrix: XGBoost", xgb_probs),
    ]:
        y_pred = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        savefig(fname)

    return ds_probs, attn, gates


# ==================================================================
# 11-12: Training diagnostics
# ==================================================================
def make_training_graphs(ctx):
    print("\n[Training diagnostic graphs]")
    d = ctx["d_taiwan"]
    n_static, n_seq_feat = ctx["n_static"], ctx["n_seq_feat"]

    # 11: Retrain briefly on full train (with an internal val split) to record
    # a fresh per-epoch loss/AUC curve for the figure. Uses model.py's
    # run_epoch directly so training math is identical to the locked model.
    scalers = make_scalers([d["X_static_train"], d["X_seq_train"]], [n_static, n_seq_feat])
    Xs_scaled, Xt_scaled = apply_scalers([d["X_static_train"], d["X_seq_train"]], scalers)
    y_train = d["y_train"]

    n = len(y_train)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    val_cut = int(n * 0.1)
    val_idx, tr_idx = perm[:val_cut], perm[val_cut:]

    train_tensors = [torch.tensor(Xs_scaled[tr_idx], dtype=torch.float32),
                      torch.tensor(Xt_scaled[tr_idx], dtype=torch.float32),
                      torch.tensor(y_train[tr_idx], dtype=torch.float32)]
    val_tensors = [torch.tensor(Xs_scaled[val_idx], dtype=torch.float32),
                    torch.tensor(Xt_scaled[val_idx], dtype=torch.float32),
                    torch.tensor(y_train[val_idx], dtype=torch.float32)]
    train_loader = make_loader(train_tensors, 256, shuffle=True)
    val_loader = make_loader(val_tensors, 256, shuffle=False)

    fresh_model = DualStreamModel(n_static, n_seq_feat).to(DEVICE)
    optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_aucs = [], []
    print("    Retraining briefly to record training-curve history (15 epochs)...")
    for epoch in range(15):
        train_metrics, _ = run_epoch(fresh_model, train_loader, optimizer, criterion, train=True, dual_stream=True)
        val_metrics, _ = run_epoch(fresh_model, val_loader, None, criterion, train=False, dual_stream=True)
        train_losses.append(train_metrics["loss"])
        val_aucs.append(val_metrics["auc"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, 16), train_losses, color="#C44E52", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(range(1, 16), val_aucs, color="#4C72B0", label="Validation AUC")
    ax2.set_ylabel("Validation AUC", color="#4C72B0")
    plt.title("Dual-Stream Model: Training Curve")
    savefig("11_dualstream_training_curve.png")

    # 12: CV AUC comparison with error bars (from results_log.json, no retraining needed)
    results = ctx["results"]
    latest_by_label = {}
    for entry in results:
        latest_by_label[entry["label"]] = entry

    taiwan_labels_order = [
        "Taiwan - Dual-Stream (Temporal GRU+Attention + Static + Gated Fusion)",
        "Taiwan - Ablation (Flattened features, no temporal structure)",
        "Taiwan - Baseline: LogisticRegression",
        "Taiwan - Baseline: RandomForest",
        "Taiwan - Baseline: XGBoost",
        "Taiwan - Baseline: SVM_RBF",
    ]
    short_names = ["Dual-Stream", "Ablation", "LogReg", "RandomForest", "XGBoost", "SVM"]
    cv_means = [latest_by_label[l]["cv_mean"]["auc"] for l in taiwan_labels_order if l in latest_by_label]
    cv_stds = [latest_by_label[l]["cv_std"]["auc"] for l in taiwan_labels_order if l in latest_by_label]
    valid_names = [n for l, n in zip(taiwan_labels_order, short_names) if l in latest_by_label]

    plt.figure(figsize=(9, 5))
    colors = ["#C44E52" if n == "Dual-Stream" else "#4C72B0" for n in valid_names]
    plt.bar(valid_names, cv_means, yerr=cv_stds, capsize=5, color=colors)
    plt.ylabel("5-Fold CV AUC (mean \u00b1 std)")
    plt.title("Cross-Validation AUC Comparison Across Models")
    plt.ylim(0.6, 0.85)
    savefig("12_cv_auc_comparison_with_error_bars.png")


# ==================================================================
# 13-16: Explainability
# ==================================================================
def make_explainability_graphs(ctx, ds_probs, attn, gates):
    print("\n[Explainability graphs]")
    y_test = ctx["y_test"]

    # 13: Attention by month
    mean_attn = attn.mean(axis=0)
    plt.figure(figsize=(8, 5))
    plt.bar(MONTH_LABELS, mean_attn, color="#4C72B0")
    plt.title("Average Temporal Attention Weight by Month")
    plt.ylabel("Attention Weight")
    plt.xticks(rotation=30, ha="right")
    savefig("13_attention_weight_by_month.png")

    # 14: Attention by outcome
    mean_attn_default = attn[y_test == 1].mean(axis=0)
    mean_attn_nodefault = attn[y_test == 0].mean(axis=0)
    x = np.arange(len(MONTH_LABELS))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, mean_attn_nodefault, width, label="No Default", color="#55A868")
    plt.bar(x + width/2, mean_attn_default, width, label="Default", color="#C44E52")
    plt.xticks(x, MONTH_LABELS, rotation=30, ha="right")
    plt.ylabel("Attention Weight")
    plt.title("Temporal Attention by True Outcome")
    plt.legend()
    savefig("14_attention_by_outcome.png")

    # 15: Gate trust distribution
    temporal_dim = ctx["dual_model"].temporal_stream.out_dim
    temporal_trust = gates[:, :temporal_dim].mean(axis=1)
    static_trust = gates[:, temporal_dim:].mean(axis=1)
    plt.figure(figsize=(8, 5))
    plt.hist(temporal_trust, bins=30, alpha=0.6, label="Temporal trust", color="#4C72B0")
    plt.hist(static_trust, bins=30, alpha=0.6, label="Static trust", color="#DD8452")
    plt.xlabel("Mean Gate Value")
    plt.ylabel("Count")
    plt.title("Distribution of Fusion-Gate Trust Scores")
    plt.legend()
    savefig("15_gate_trust_distribution.png")

    # 16: SHAP feature importance
    import shap
    n_static, n_seq_feat = ctx["n_static"], ctx["n_seq_feat"]
    n_timesteps = ctx["Xt_test"].shape[1]

    class FlatWrapper(nn.Module):
        def __init__(self, model, n_static, seq_shape):
            super().__init__()
            self.model = model
            self.n_static = n_static
            self.seq_shape = seq_shape

        def forward(self, x_flat):
            x_static = x_flat[:, :self.n_static]
            x_seq = x_flat[:, self.n_static:].reshape(-1, *self.seq_shape)
            logits, _, _ = self.model(x_static, x_seq)
            return torch.sigmoid(logits).unsqueeze(-1)

    feature_names = list(STATIC_FEATURE_NAMES)
    for t in range(n_timesteps):
        for ch in SEQ_CHANNEL_NAMES:
            feature_names.append(f"{ch}_month{t+1}")

    X_flat_test = np.hstack([ctx["Xs_test"], ctx["Xt_test"].reshape(len(ctx["Xs_test"]), -1)]).astype(np.float32)
    wrapper = FlatWrapper(ctx["dual_model"], n_static, (n_timesteps, n_seq_feat)).to(DEVICE)
    wrapper.eval()

    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_flat_test), size=min(100, len(X_flat_test)), replace=False)
    explain_idx = rng.choice(len(X_flat_test), size=min(200, len(X_flat_test)), replace=False)
    background = torch.tensor(X_flat_test[bg_idx], dtype=torch.float32).to(DEVICE)
    explain_data = torch.tensor(X_flat_test[explain_idx], dtype=torch.float32).to(DEVICE)

    print("    Running SHAP GradientExplainer (~1 min)...")
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = np.array(explainer.shap_values(explain_data))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:15]

    plt.figure(figsize=(9, 7))
    plt.barh([feature_names[i] for i in order][::-1], mean_abs_shap[order][::-1], color="#4C72B0")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 15 Features by SHAP Importance (Dual-Stream Model)")
    savefig("16_shap_feature_importance.png")


# ==================================================================
# 17: Generalization comparison
# ==================================================================
def make_generalization_graph(ctx):
    print("\n[Generalization graph]")
    results = ctx["results"]
    latest_by_label = {}
    for entry in results:
        latest_by_label[entry["label"]] = entry

    pairs = [
        ("Taiwan - Ablation (Flattened features, no temporal structure)", "Taiwan\n(Static-Only)"),
        ("German - Static-Only (Generalization Check)", "German\n(Static-Only)"),
    ]
    names = [p[1] for p in pairs if p[0] in latest_by_label]
    aucs = [latest_by_label[p[0]]["test"]["auc"] for p in pairs if p[0] in latest_by_label]

    plt.figure(figsize=(6, 5))
    plt.bar(names, aucs, color=["#4C72B0", "#DD8452"])
    plt.ylabel("Test AUC")
    plt.title("Static-Only Model: Generalization Across Datasets")
    plt.ylim(0, 1)
    for i, v in enumerate(aucs):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center")
    savefig("17_dataset_generalization_comparison.png")


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    ctx = load_all()

    make_eda_graphs(ctx)
    ds_probs, attn, gates = make_performance_graphs(ctx)
    make_training_graphs(ctx)
    make_explainability_graphs(ctx, ds_probs, attn, gates)
    make_generalization_graph(ctx)

    print("\n" + "=" * 70)
    print(f"  ALL 17 GRAPHS GENERATED -> '{OUTPUT_DIR}/'")
    print("=" * 70)
