"""
verify_results_log.py
------------------------
Checks that results_log.json contains all 12 expected experiment entries
(3 from model.py, 8 from baseline_comparison.py, 1 from ensemble_check.py),
and prints a clean summary table. Flags anything missing.

Run from PowerShell:
    cd "D:\MITS\08-09-26\Dataset"
    python verify_results_log.py
"""

import json
import os

RESULTS_LOG = "results_log.json"

EXPECTED_LABELS = [
    "Taiwan - Dual-Stream (Temporal GRU+Attention + Static + Gated Fusion)",
    "Taiwan - Ablation (Flattened features, no temporal structure)",
    "German - Static-Only (Generalization Check)",
    "Taiwan - Baseline: LogisticRegression",
    "Taiwan - Baseline: RandomForest",
    "Taiwan - Baseline: XGBoost",
    "Taiwan - Baseline: SVM_RBF",
    "German - Baseline: LogisticRegression",
    "German - Baseline: RandomForest",
    "German - Baseline: XGBoost",
    "German - Baseline: SVM_RBF",
]
# Ensemble label includes a variable weight, so match by prefix
ENSEMBLE_PREFIX = "Taiwan - Ensemble"

if __name__ == "__main__":
    if not os.path.exists(RESULTS_LOG):
        print(f"[!] {RESULTS_LOG} not found in this folder.")
        exit(1)

    with open(RESULTS_LOG, "r") as f:
        results = json.load(f)

    print(f"Total entries in {RESULTS_LOG}: {len(results)}\n")

    found_labels = set(r["label"] for r in results)
    has_ensemble = any(r["label"].startswith(ENSEMBLE_PREFIX) for r in results)

    print(f"{'Status':8s} {'Label':70s} {'Test AUC':>10s} {'Test F1':>10s}")
    print("-" * 100)

    missing = []
    for label in EXPECTED_LABELS:
        matches = [r for r in results if r["label"] == label]
        if matches:
            r = matches[-1]  # latest if duplicated
            print(f"{'[OK]':8s} {label:70s} {r['test']['auc']:10.4f} {r['test']['f1']:10.4f}")
        else:
            print(f"{'[MISS]':8s} {label:70s} {'--':>10s} {'--':>10s}")
            missing.append(label)

    ensemble_entries = [r for r in results if r["label"].startswith(ENSEMBLE_PREFIX)]
    if ensemble_entries:
        r = ensemble_entries[-1]
        print(f"{'[OK]':8s} {r['label']:70s} {r['test']['auc']:10.4f} {r['test']['f1']:10.4f}")
    else:
        print(f"{'[MISS]':8s} {'Taiwan - Ensemble (...)':70s} {'--':>10s} {'--':>10s}")
        missing.append("Taiwan - Ensemble")

    print("\n" + "=" * 70)
    if not missing:
        print(f"  ALL {len(EXPECTED_LABELS) + 1} EXPECTED ENTRIES PRESENT. Results log is complete.")
    else:
        print(f"  MISSING {len(missing)} ENTRIES:")
        for m in missing:
            print(f"    - {m}")
        print("\n  Re-run the corresponding script(s) to fill these in:")
        print("    model.py               -> Dual-Stream, Ablation, German Static-Only")
        print("    baseline_comparison.py -> all 8 baseline entries")
        print("    ensemble_check.py      -> Ensemble entry")
    print("=" * 70)
