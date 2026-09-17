"""
model.py
---------
*** LOCKED ARCHITECTURE - v1.0 ***
Do not modify TemporalStream / StaticStream / GatedFusion / DualStreamModel
structure below without incrementing MODEL_VERSION. Hyperparameter tuning
(hidden dim, lr, epochs, patience) is fine; structural changes are not,
since results must stay comparable to the results log this script writes.

Dual-Stream Temporal-Static Hybrid Network for credit default prediction.

Architecture:
    Stream 1 (temporal): GRU over 6-month sequence -> self-attention over timesteps
    Stream 2 (static):   Dense encoder over static/demographic features
    Fusion:              Learned gate combines both stream representations
    Head:                Dense classifier -> default probability

Experimental setup:
    Train (80%) -> 5-Fold Stratified CV (reported as mean +/- std)
    Final model retrained on full Train -> evaluated once on held-out Test

Also runs a static-only version of the model on the German dataset as a
generalization check (no temporal stream available there).

Run from PowerShell (after running preprocess_datasets.py):
    cd "D:\MITS\08-09-26\Dataset"
    python model.py

Requires: torch, numpy, scikit-learn
    pip install torch numpy scikit-learn

Outputs:
    checkpoints/<experiment>_final.pt   - trained weights for each experiment
    results_log.json                    - CV + test metrics for every run,
                                           appended with a timestamp so nothing
                                           gets overwritten silently
"""

import os
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "1.1"  # bumped: fixed row-level dedup leakage + per-fold scaling
CHECKPOINT_DIR = "checkpoints"
RESULTS_LOG = "results_log.json"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PROCESSED_DIR = "processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
PATIENCE = 5          # early stopping patience
HIDDEN_DIM = 32

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ==================================================================
# MODEL DEFINITION
# ==================================================================
class TemporalStream(nn.Module):
    """GRU + self-attention over the 6-month sequence."""
    def __init__(self, n_features, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.out_dim = hidden_dim

    def forward(self, x_seq):
        # x_seq: (B, T, F)
        gru_out, _ = self.gru(x_seq)              # (B, T, H)
        scores = self.attn_score(gru_out)          # (B, T, 1)
        attn_weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = (gru_out * attn_weights).sum(dim=1)  # (B, H)
        return context, attn_weights.squeeze(-1)    # attn_weights: (B, T) for explainability


class StaticStream(nn.Module):
    """Dense encoder over static/demographic features."""
    def __init__(self, n_features, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x_static):
        return self.net(x_static)


class GatedFusion(nn.Module):
    """Learns how much to trust the temporal vs. static representation per-instance."""
    def __init__(self, temporal_dim, static_dim):
        super().__init__()
        combined_dim = temporal_dim + static_dim
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Sigmoid()
        )
        self.out_dim = combined_dim

    def forward(self, temporal_repr, static_repr):
        combined = torch.cat([temporal_repr, static_repr], dim=1)
        gate_values = self.gate(combined)           # (B, combined_dim), values in [0,1]
        fused = gate_values * combined
        return fused, gate_values


class DualStreamModel(nn.Module):
    """Full model: temporal stream + static stream -> gated fusion -> classifier."""
    def __init__(self, n_static_features, n_seq_features, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.temporal_stream = TemporalStream(n_seq_features, hidden_dim)
        self.static_stream = StaticStream(n_static_features, hidden_dim)
        self.fusion = GatedFusion(self.temporal_stream.out_dim, self.static_stream.out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_static, x_seq):
        temporal_repr, attn_weights = self.temporal_stream(x_seq)
        static_repr = self.static_stream(x_static)
        fused, gate_values = self.fusion(temporal_repr, static_repr)
        logit = self.classifier(fused).squeeze(-1)
        return logit, attn_weights, gate_values


class StaticOnlyModel(nn.Module):
    """Static-stream-only variant, used for the German generalization check
    and as the flattened-feature ablation baseline on Taiwan."""
    def __init__(self, n_static_features, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.static_stream = StaticStream(n_static_features, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.static_stream.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_static):
        static_repr = self.static_stream(x_static)
        logit = self.classifier(static_repr).squeeze(-1)
        return logit


# ==================================================================
# SHARED TRAIN / EVAL LOGIC  (used by every fold, final fit, and test)
# ==================================================================
def make_loader(tensors, batch_size, shuffle):
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def run_epoch(model, loader, optimizer, criterion, train, dual_stream):
    model.train() if train else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if dual_stream:
                xb_static, xb_seq, yb = [t.to(DEVICE) for t in batch]
                logits, _, _ = model(xb_static, xb_seq)
            else:
                xb, yb = [t.to(DEVICE) for t in batch]
                logits = model(xb)

            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(y_true)
    return metrics, y_prob


def train_with_early_stopping(model, train_loader, val_loader, dual_stream, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    """Single reusable training loop. Used for every CV fold and the final fit
    so training logic is never duplicated."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc, best_state, epochs_no_improve = -np.inf, None, 0

    for epoch in range(1, epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, criterion, train=True, dual_stream=dual_stream)
        val_metrics, _ = run_epoch(model, val_loader, optimizer, criterion, train=False, dual_stream=dual_stream)

        improved = val_metrics["auc"] > best_val_auc
        if improved:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"    Epoch {epoch:2d} | train_loss={train_metrics['loss']:.4f} "
                  f"val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f}"
                  f"{'  <- best' if improved else ''}")

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return model, best_val_auc


def summarize_cv(fold_metrics_list, label):
    print(f"\n[CV Summary] {label} (mean +/- std over {len(fold_metrics_list)} folds)")
    for key in ["auc", "f1", "accuracy", "precision", "recall"]:
        values = [m[key] for m in fold_metrics_list]
        print(f"    {key:10s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


# ==================================================================
# EXPERIMENT RUNNER: 5-Fold CV -> Final Train -> Test
# (one function, reused for both the dual-stream Taiwan model and the
#  static-only German model, so the experimental logic is never repeated)
# ==================================================================
def make_scalers(train_arrays, n_numeric_per_array):
    """Fits one StandardScaler per input array using ONLY the given rows
    (the caller passes the fold's training subset here -- never validation
    or test rows), so there is no statistical leakage into what gets scaled.
    For arrays with n_numeric < total width (e.g. German's numeric+one-hot
    matrix), only the first n_numeric columns are scaled; one-hot columns
    are left untouched. 3D sequence arrays are scaled by flattening the
    timestep dimension first, then reshaping back."""
    scalers = []
    for arr, n_num in zip(train_arrays, n_numeric_per_array):
        if arr.ndim == 3:
            n, t, f = arr.shape
            flat = arr.reshape(-1, f)
            n_num = n_num if n_num is not None else f
            scaler = StandardScaler().fit(flat[:, :n_num])
            scalers.append(("seq", scaler, n_num))
        else:
            n_num = n_num if n_num is not None else arr.shape[1]
            scaler = StandardScaler().fit(arr[:, :n_num])
            scalers.append(("flat", scaler, n_num))
    return scalers


def apply_scalers(arrays, scalers):
    """Applies already-fitted scalers to any set of arrays (train, val, or test)."""
    out = []
    for arr, (kind, scaler, n_num) in zip(arrays, scalers):
        if kind == "seq":
            n, t, f = arr.shape
            flat = arr.reshape(-1, f).astype(np.float32).copy()
            flat[:, :n_num] = scaler.transform(flat[:, :n_num])
            out.append(flat.reshape(n, t, f))
        else:
            arr = arr.astype(np.float32).copy()
            arr[:, :n_num] = scaler.transform(arr[:, :n_num])
            out.append(arr)
    return out


def log_results(label, fold_metrics_list, test_metrics, model_version=None):
    """Appends this run's results to results_log.json, tagged with the model
    version and a timestamp, so every locked-model run is kept on record.
    Also stores the raw per-fold metrics (cv_folds) -- not just mean/std --
    so a later statistical significance test (e.g. paired Wilcoxon across
    folds) can be run without needing to retrain anything."""
    entry = {
        "label": label,
        "model_version": model_version if model_version is not None else MODEL_VERSION,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "cv_mean": {k: float(np.mean([m[k] for m in fold_metrics_list])) for k in
                    ["auc", "f1", "accuracy", "precision", "recall"]},
        "cv_std": {k: float(np.std([m[k] for m in fold_metrics_list])) for k in
                   ["auc", "f1", "accuracy", "precision", "recall"]},
        "cv_folds": [
            {k: float(m[k]) for k in ["auc", "f1", "accuracy", "precision", "recall"]}
            for m in fold_metrics_list
        ],
        "test": {k: float(test_metrics[k]) for k in
                 ["auc", "f1", "accuracy", "precision", "recall"]},
    }
    history = []
    if os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(entry)
    with open(RESULTS_LOG, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Logged] Results appended to {RESULTS_LOG}")


def run_experiment(label, build_model_fn, X_train_tuple, y_train, fold_train_idx, fold_val_idx,
                    X_test_tuple, y_test, dual_stream, n_numeric_per_array=None):
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {label}")
    print("=" * 70)

    if n_numeric_per_array is None:
        n_numeric_per_array = [None] * len(X_train_tuple)

    # ---------- 5-Fold Stratified Cross-Validation ----------
    fold_metrics_list = []
    for fold_i, (tr_idx, val_idx) in enumerate(zip(fold_train_idx, fold_val_idx), start=1):
        print(f"\n  -- Fold {fold_i}/{len(fold_train_idx)} --")
        tr_idx, val_idx = np.array(tr_idx, dtype=int), np.array(val_idx, dtype=int)

        # Fit scaler on THIS FOLD's training rows only -- never on val_idx.
        raw_tr_arrays = [x[tr_idx] for x in X_train_tuple]
        raw_val_arrays = [x[val_idx] for x in X_train_tuple]
        scalers = make_scalers(raw_tr_arrays, n_numeric_per_array)
        scaled_tr_arrays = apply_scalers(raw_tr_arrays, scalers)
        scaled_val_arrays = apply_scalers(raw_val_arrays, scalers)

        train_tensors = [torch.tensor(x, dtype=torch.float32) for x in scaled_tr_arrays]
        train_tensors.append(torch.tensor(y_train[tr_idx], dtype=torch.float32))
        val_tensors = [torch.tensor(x, dtype=torch.float32) for x in scaled_val_arrays]
        val_tensors.append(torch.tensor(y_train[val_idx], dtype=torch.float32))

        train_loader = make_loader(train_tensors, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_tensors, BATCH_SIZE, shuffle=False)

        model = build_model_fn()
        model, _ = train_with_early_stopping(model, train_loader, val_loader, dual_stream)

        val_metrics, _ = run_epoch(model, val_loader, None, nn.BCEWithLogitsLoss(), train=False, dual_stream=dual_stream)
        fold_metrics_list.append(val_metrics)

    summarize_cv(fold_metrics_list, label)

    # ---------- Final fit on FULL training set, evaluate once on TEST ----------
    print(f"\n  -- Final model: training on full train set, evaluating on held-out test --")

    # Fit the final scaler on the FULL training set only (still zero test leakage),
    # then apply it to both train and test.
    final_scalers = make_scalers(list(X_train_tuple), n_numeric_per_array)
    scaled_full_train = apply_scalers(list(X_train_tuple), final_scalers)
    scaled_test = apply_scalers(list(X_test_tuple), final_scalers)

    full_train_tensors = [torch.tensor(x, dtype=torch.float32) for x in scaled_full_train]
    full_train_tensors.append(torch.tensor(y_train, dtype=torch.float32))
    test_tensors = [torch.tensor(x, dtype=torch.float32) for x in scaled_test]
    test_tensors.append(torch.tensor(y_test, dtype=torch.float32))

    # carve a small validation slice out of train for early stopping on the final fit
    n = len(y_train)
    val_cut = int(n * 0.1)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    final_val_idx, final_train_idx = perm[:val_cut], perm[val_cut:]

    final_train_tensors = [t[final_train_idx] for t in full_train_tensors[:-1]] + [full_train_tensors[-1][final_train_idx]]
    final_val_tensors = [t[final_val_idx] for t in full_train_tensors[:-1]] + [full_train_tensors[-1][final_val_idx]]

    final_train_loader = make_loader(final_train_tensors, BATCH_SIZE, shuffle=True)
    final_val_loader = make_loader(final_val_tensors, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_tensors, BATCH_SIZE, shuffle=False)

    final_model = build_model_fn()
    final_model, _ = train_with_early_stopping(final_model, final_train_loader, final_val_loader, dual_stream)

    test_metrics, test_probs = run_epoch(final_model, test_loader, None, nn.BCEWithLogitsLoss(), train=False, dual_stream=dual_stream)
    print(f"\n[TEST RESULTS] {label}")
    for key in ["auc", "f1", "accuracy", "precision", "recall"]:
        print(f"    {key:10s}: {test_metrics[key]:.4f}")

    # ---------- Save checkpoint + log results (locked-model record-keeping) ----------
    safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("-", "").replace("+", "plus")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{safe_name}_final_v{MODEL_VERSION}.pt")
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"[Saved] Checkpoint -> {ckpt_path}")
    log_results(label, fold_metrics_list, test_metrics)

    return final_model, fold_metrics_list, test_metrics


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # ---------------- Taiwan: Dual-Stream model ----------------
    taiwan_path = os.path.join(PROCESSED_DIR, "taiwan_processed.npz")
    if os.path.exists(taiwan_path):
        d = np.load(taiwan_path, allow_pickle=True)
        n_static = d["X_static_train"].shape[1]
        n_seq_feat = d["X_seq_train"].shape[2]

        run_experiment(
            label="Taiwan - Dual-Stream (Temporal GRU+Attention + Static + Gated Fusion)",
            build_model_fn=lambda: DualStreamModel(n_static, n_seq_feat),
            X_train_tuple=(d["X_static_train"], d["X_seq_train"]),
            y_train=d["y_train"],
            fold_train_idx=d["fold_train_idx"],
            fold_val_idx=d["fold_val_idx"],
            X_test_tuple=(d["X_static_test"], d["X_seq_test"]),
            y_test=d["y_test"],
            dual_stream=True,
            n_numeric_per_array=[n_static, n_seq_feat],  # all columns numeric in both streams
        )

        # ---------------- Taiwan: Flattened-feature ablation baseline ----------------
        # Same static features + the temporal data averaged into static form (no sequence structure)
        # -> proves whether the temporal stream is actually adding value
        seq_train_flat = d["X_seq_train"].mean(axis=1)   # (N, 3) - collapse the 6 timesteps
        seq_test_flat = d["X_seq_test"].mean(axis=1)
        X_static_train_ablation = np.hstack([d["X_static_train"], seq_train_flat])
        X_static_test_ablation = np.hstack([d["X_static_test"], seq_test_flat])
        n_static_ablation = X_static_train_ablation.shape[1]

        run_experiment(
            label="Taiwan - Ablation (Flattened features, no temporal structure)",
            build_model_fn=lambda: StaticOnlyModel(n_static_ablation),
            X_train_tuple=(X_static_train_ablation,),
            y_train=d["y_train"],
            fold_train_idx=d["fold_train_idx"],
            fold_val_idx=d["fold_val_idx"],
            X_test_tuple=(X_static_test_ablation,),
            y_test=d["y_test"],
            dual_stream=False,
            n_numeric_per_array=[n_static_ablation],  # all columns numeric
        )
    else:
        print(f"[!] {taiwan_path} not found. Run preprocess_datasets.py first.")

    # ---------------- German: Static-only generalization check ----------------
    german_path = os.path.join(PROCESSED_DIR, "german_processed.npz")
    if os.path.exists(german_path):
        d = np.load(german_path, allow_pickle=True)
        n_feat = d["X_train"].shape[1]
        n_numeric_german = int(d["n_numeric"])  # only these columns get scaled; rest are one-hot

        run_experiment(
            label="German - Static-Only (Generalization Check)",
            build_model_fn=lambda: StaticOnlyModel(n_feat),
            X_train_tuple=(d["X_train"],),
            y_train=d["y_train"],
            fold_train_idx=d["fold_train_idx"],
            fold_val_idx=d["fold_val_idx"],
            X_test_tuple=(d["X_test"],),
            y_test=d["y_test"],
            dual_stream=False,
            n_numeric_per_array=[n_numeric_german],  # one-hot columns left unscaled
        )
    else:
        print(f"[!] {german_path} not found. Run preprocess_datasets.py first.")

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
