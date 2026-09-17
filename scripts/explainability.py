"""
explainability.py
--------------------
Generates the explainability analysis + figures for the locked Dual-Stream
model (v1.1), using the held-out test set:

    1. Temporal attention weights -- which of the 6 months drove each
       prediction (from the model's built-in self-attention layer).
    2. Fusion-gate values -- how much the model trusted the temporal
       representation vs. the static representation, per prediction.
    3. SHAP values -- feature-level attribution on the flattened
       representation (static features + 18 raw temporal values), using
       a wrapper around the trained model so SHAP can treat it as a
       single-input function.

Reuses model.py's DualStreamModel, make_scalers, and apply_scalers directly
-- no logic is duplicated.

Run from PowerShell (same folder as model.py and the checkpoints/ +
processed/ directories produced by earlier steps):
    cd "D:\MITS\08-09-26\Dataset"
    pip install shap matplotlib seaborn
    python explainability.py

Outputs (saved to explainability_output/):
    attention_by_month.png       - average attention weight per month
    attention_by_outcome.png     - attention pattern split by true outcome
    gate_trust_distribution.png  - temporal vs static trust score histogram
    shap_feature_importance.png  - mean |SHAP value| per feature
    explainability_summary.json  - all the numeric results, for the paper
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import DualStreamModel, make_scalers, apply_scalers, RANDOM_STATE

PROCESSED_DIR = "processed"
CHECKPOINT_PATH = os.path.join(
    "checkpoints",
    "taiwan__dualstream_temporal_gruplusattention_plus_static_plus_gated_fusion_final_v1.1.pt"
)
OUTPUT_DIR = "explainability_output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SHAP_SAMPLES = 200     # test rows to explain with SHAP (kept modest for speed)
N_SHAP_BACKGROUND = 100  # background rows SHAP uses as its reference distribution

MONTH_LABELS = ["Month t-5", "Month t-4", "Month t-3", "Month t-2", "Month t-1", "Month t (most recent)"]
STATIC_FEATURE_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                         "UTILIZATION_RATIO", "PAYMENT_RATIO", "MAX_DELINQUENCY"]
SEQ_CHANNEL_NAMES = ["PAY_STATUS", "BILL_AMT", "PAY_AMT"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_model_and_data():
    d = np.load(os.path.join(PROCESSED_DIR, "taiwan_processed.npz"), allow_pickle=True)
    n_static = d["X_static_train"].shape[1]
    n_seq_feat = d["X_seq_train"].shape[2]

    # Reproduce the exact same scaling the final model was trained with:
    # fit on the full training set, apply to test (see model.py run_experiment).
    scalers = make_scalers([d["X_static_train"], d["X_seq_train"]], [n_static, n_seq_feat])
    Xs_test, Xt_test = apply_scalers([d["X_static_test"], d["X_seq_test"]], scalers)
    y_test = d["y_test"]

    model = DualStreamModel(n_static, n_seq_feat)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, Xs_test, Xt_test, y_test, n_static, n_seq_feat


# ==================================================================
# PART 1 & 2: Attention weights + Gate values (direct forward pass)
# ==================================================================
def extract_attention_and_gates(model, Xs_test, Xt_test, y_test, batch_size=512):
    all_attn, all_gates, all_probs = [], [], []
    n = len(y_test)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xs = torch.tensor(Xs_test[start:end], dtype=torch.float32).to(DEVICE)
            xt = torch.tensor(Xt_test[start:end], dtype=torch.float32).to(DEVICE)
            logits, attn_weights, gate_values = model(xs, xt)
            all_attn.append(attn_weights.cpu().numpy())
            all_gates.append(gate_values.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    attn = np.concatenate(all_attn, axis=0)      # (N, 6)
    gates = np.concatenate(all_gates, axis=0)    # (N, temporal_dim + static_dim)
    probs = np.concatenate(all_probs, axis=0)    # (N,)
    return attn, gates, probs


def analyze_attention(attn, y_test, probs):
    print("\n" + "=" * 70)
    print("  ATTENTION ANALYSIS (which month drove the decision)")
    print("=" * 70)

    mean_attn_overall = attn.mean(axis=0)
    for label, val in zip(MONTH_LABELS, mean_attn_overall):
        print(f"    {label:25s}: {val:.4f}")

    # Split by true outcome
    mean_attn_default = attn[y_test == 1].mean(axis=0)
    mean_attn_nodefault = attn[y_test == 0].mean(axis=0)

    # Plot 1: overall average attention per month
    plt.figure(figsize=(8, 5))
    plt.bar(MONTH_LABELS, mean_attn_overall, color="#4C72B0")
    plt.title("Average Temporal Attention Weight by Month (Test Set)")
    plt.ylabel("Attention Weight")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "attention_by_month.png"))
    plt.close()

    # Plot 2: attention pattern split by true outcome
    x = np.arange(len(MONTH_LABELS))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, mean_attn_nodefault, width, label="No Default (y=0)", color="#55A868")
    plt.bar(x + width/2, mean_attn_default, width, label="Default (y=1)", color="#C44E52")
    plt.xticks(x, MONTH_LABELS, rotation=30, ha="right")
    plt.ylabel("Attention Weight")
    plt.title("Temporal Attention by True Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "attention_by_outcome.png"))
    plt.close()

    most_attended_idx = int(np.argmax(mean_attn_overall))
    print(f"\n    Most-attended month overall: {MONTH_LABELS[most_attended_idx]} "
          f"(weight={mean_attn_overall[most_attended_idx]:.4f})")

    return {
        "mean_attention_overall": mean_attn_overall.tolist(),
        "mean_attention_default": mean_attn_default.tolist(),
        "mean_attention_no_default": mean_attn_nodefault.tolist(),
        "most_attended_month": MONTH_LABELS[most_attended_idx],
    }


def analyze_gates(gates, y_test, probs, temporal_dim=32):
    print("\n" + "=" * 70)
    print("  FUSION GATE ANALYSIS (temporal vs static trust)")
    print("=" * 70)

    temporal_trust = gates[:, :temporal_dim].mean(axis=1)  # per-sample avg gate value, temporal half
    static_trust = gates[:, temporal_dim:].mean(axis=1)    # per-sample avg gate value, static half

    print(f"    Mean temporal trust (all test samples): {temporal_trust.mean():.4f}")
    print(f"    Mean static trust   (all test samples): {static_trust.mean():.4f}")

    trust_default = temporal_trust[y_test == 1].mean()
    trust_nodefault = temporal_trust[y_test == 0].mean()
    print(f"    Mean temporal trust for DEFAULT cases:    {trust_default:.4f}")
    print(f"    Mean temporal trust for NO-DEFAULT cases: {trust_nodefault:.4f}")

    plt.figure(figsize=(8, 5))
    plt.hist(temporal_trust, bins=30, alpha=0.6, label="Temporal trust", color="#4C72B0")
    plt.hist(static_trust, bins=30, alpha=0.6, label="Static trust", color="#DD8452")
    plt.xlabel("Mean Gate Value")
    plt.ylabel("Count")
    plt.title("Distribution of Fusion-Gate Trust Scores (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gate_trust_distribution.png"))
    plt.close()

    return {
        "mean_temporal_trust": float(temporal_trust.mean()),
        "mean_static_trust": float(static_trust.mean()),
        "temporal_trust_default_cases": float(trust_default),
        "temporal_trust_no_default_cases": float(trust_nodefault),
    }


# ==================================================================
# PART 3: SHAP feature attribution
# ==================================================================
class FlatWrapper(nn.Module):
    """Wraps DualStreamModel so SHAP can treat it as a single-input function:
    takes one flattened tensor (static features + flattened 6x3 sequence),
    reshapes internally, and returns the default probability."""
    def __init__(self, model, n_static, seq_shape):
        super().__init__()
        self.model = model
        self.n_static = n_static
        self.seq_shape = seq_shape  # (timesteps, channels)

    def forward(self, x_flat):
        x_static = x_flat[:, :self.n_static]
        x_seq = x_flat[:, self.n_static:].reshape(-1, *self.seq_shape)
        logits, _, _ = self.model(x_static, x_seq)
        return torch.sigmoid(logits).unsqueeze(-1)


def run_shap_analysis(model, Xs_test, Xt_test, n_static, n_seq_feat):
    print("\n" + "=" * 70)
    print("  SHAP FEATURE ATTRIBUTION")
    print("=" * 70)

    import shap

    n_timesteps = Xt_test.shape[1]
    X_flat_test = np.hstack([Xs_test, Xt_test.reshape(len(Xs_test), -1)]).astype(np.float32)

    feature_names = list(STATIC_FEATURE_NAMES)
    for t in range(n_timesteps):
        for ch in SEQ_CHANNEL_NAMES:
            feature_names.append(f"{ch}_month{t+1}")

    wrapper = FlatWrapper(model, n_static, (n_timesteps, n_seq_feat)).to(DEVICE)
    wrapper.eval()

    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_flat_test), size=min(N_SHAP_BACKGROUND, len(X_flat_test)), replace=False)
    explain_idx = rng.choice(len(X_flat_test), size=min(N_SHAP_SAMPLES, len(X_flat_test)), replace=False)

    background = torch.tensor(X_flat_test[bg_idx], dtype=torch.float32).to(DEVICE)
    explain_data = torch.tensor(X_flat_test[explain_idx], dtype=torch.float32).to(DEVICE)

    print(f"    Running SHAP GradientExplainer on {len(explain_idx)} samples "
          f"(background={len(bg_idx)})... this may take a minute.")
    # GradientExplainer (expected gradients) is used instead of DeepExplainer
    # because DeepExplainer's DeepLIFT rules don't reliably support GRU layers
    # (raises an "unrecognized nn.Module: GRU" warning and can fail its
    # additivity check). GradientExplainer works directly with autograd and
    # handles arbitrary differentiable architectures, including RNNs.
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(explain_data)

    # shap_values shape: (N, n_features, 1) for a single-output model -- squeeze last dim
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    top_n = 15

    print(f"\n    Top {top_n} features by mean |SHAP value|:")
    for i in order[:top_n]:
        print(f"    {feature_names[i]:25s}: {mean_abs_shap[i]:.5f}")

    plt.figure(figsize=(9, 7))
    top_idx = order[:top_n]
    plt.barh([feature_names[i] for i in top_idx][::-1], mean_abs_shap[top_idx][::-1], color="#4C72B0")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} Features by SHAP Importance (Dual-Stream Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_feature_importance.png"))
    plt.close()

    return {
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs_shap.tolist(),
        "top_features": [feature_names[i] for i in order[:top_n]],
    }


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    model, Xs_test, Xt_test, y_test, n_static, n_seq_feat = load_model_and_data()

    attn, gates, probs = extract_attention_and_gates(model, Xs_test, Xt_test, y_test)
    attention_results = analyze_attention(attn, y_test, probs)
    gate_results = analyze_gates(gates, y_test, probs, temporal_dim=model.temporal_stream.out_dim)
    shap_results = run_shap_analysis(model, Xs_test, Xt_test, n_static, n_seq_feat)

    summary = {
        "attention": attention_results,
        "gates": gate_results,
        "shap": shap_results,
    }
    summary_path = os.path.join(OUTPUT_DIR, "explainability_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"  Figures + summary saved to '{OUTPUT_DIR}/'")
    print("=" * 70)
