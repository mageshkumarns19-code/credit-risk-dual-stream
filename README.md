# Dual-Stream Temporal-Static Hybrid Network for Credit Default Prediction

Code accompanying the manuscript *"A Dual-Stream Temporal-Static Hybrid Network with Gated Fusion for Explainable Credit Default Prediction."*

A GRU-with-self-attention temporal stream and a dense static stream are combined through a learned, per-instance sigmoid gate, rather than fixed concatenation, for credit default prediction. The architecture is validated on the Taiwan Credit Card Default dataset (leakage-audited train/5-fold CV/test protocol) against four baselines and a flattened-feature ablation, with an independent generalization check on the German Credit dataset. Explainability is provided natively (attention weights, fusion-gate values) and post-hoc (SHAP), with all three methods converging on the same finding: the most recent month of repayment behavior is the dominant predictive signal.

## Repository structure

```
.
├── scripts/                       All pipeline code (flat directory -- see note below)
│   ├── explore_datasets.py            Step 0: EDA on raw datasets
│   ├── preprocess_datasets.py         Step 1: cleaning, dedup, feature engineering, CV folds
│   ├── model.py                       Step 2: Dual-Stream model definition + training/eval
│   ├── baseline_comparison.py         Step 3: LR / RF / XGBoost / SVM baselines
│   ├── ensemble_check.py              Step 4: Dual-Stream + XGBoost blend (out-of-fold)
│   ├── explainability.py              Step 5: attention / gate / SHAP analysis
│   ├── generate_graphs.py             Step 6: all study figures
│   ├── generate_architecture_diagram.py   Step 6b: architecture diagram figure
│   ├── export_tables.py               Step 7: CSV tables for the manuscript
│   ├── significance_test.py           Step 8: paired Wilcoxon + t-test vs. baselines
│   └── verify_results_log.py          Utility: confirms all expected results were logged
├── requirements.txt
└── .gitignore
```

**Note on imports:** several scripts import directly from one another (e.g. `baseline_comparison.py` imports `log_results` from `model.py`, `ensemble_check.py` imports from both). All scripts therefore live in one flat `scripts/` directory and must be run with `scripts/` as the working directory, not as a package.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets

Datasets are not included in this repository. Download them and place the CSVs in `scripts/`:

- **Taiwan Credit Card Default** (`UCI_Credit_Card.csv`) -- UCI Machine Learning Repository, Yeh & Lien (2009)
- **German Credit / Statlog** (`german_credit.csv`) -- UCI Machine Learning Repository, Hofmann (1994), https://doi.org/10.24432/C5NC77

`preprocess_datasets.py` expects the raw Taiwan file with its original UCI column names, and a cleaned German credit CSV with columns as described in `scripts/preprocess_datasets.py`.

## Running the pipeline

All commands are run from inside `scripts/`, in this order:

```bash
cd scripts

# 1. Preprocess: dedup, clean categories, engineer features, build CV folds
python preprocess_datasets.py

# 2. Train the Dual-Stream model + flattened-feature ablation + German check
python model.py

# 3. Train the four baselines (Logistic Regression, Random Forest, XGBoost, SVM)
python baseline_comparison.py

# 4. Ensemble check (Dual-Stream + XGBoost, leakage-safe out-of-fold blending)
python ensemble_check.py

# 5. Explainability: attention weights, fusion-gate values, SHAP attribution
python explainability.py

# 6. Generate all study figures (EDA, performance, training curves, XAI)
python generate_graphs.py
python generate_architecture_diagram.py

# 7. Export result tables (CSV) for the manuscript
python export_tables.py

# 8. Statistical significance testing (paired Wilcoxon + t-test)
python significance_test.py

# Optional: confirm every expected experiment was logged
python verify_results_log.py
```

Each script is idempotent and can be re-run independently once its inputs exist; `results_log.json` accumulates results across runs rather than overwriting them, so delete it first if you want a clean run.

## Outputs (generated locally, not tracked in git)

- `processed/` -- preprocessed `.npz` files (train/test splits, CV fold indices)
- `checkpoints/` -- trained model weights (`.pt`)
- `results_log.json` -- every experiment's CV and test metrics, versioned and timestamped
- `graphs_output/` -- all figures (18 total, numbered)
- `tables_output/` -- all result tables (7 total, CSV)
- `explainability_output/` -- attention/gate/SHAP figures and a JSON summary

## Model summary

| Component | Configuration |
|---|---|
| Temporal stream | GRU (hidden size 32) + additive self-attention over 6 monthly timesteps |
| Static stream | 2-layer dense encoder (32 units, ReLU, dropout 0.2) |
| Fusion | Learned sigmoid gate over concatenated [temporal; static] representation |
| Classifier head | Dense(32, ReLU) \u2192 Dropout \u2192 Dense(1) |
| Training | Adam, lr=1e-3, batch size 256, max 30 epochs, early stopping (patience 5, monitored on validation AUC) |
| Evaluation | 80/20 stratified train/test split, 5-fold stratified CV on the training set only |

Full architectural detail, equations, and pseudocode are described in the accompanying manuscript.

## License

See `LICENSE`.

## Citation

If you use this code, please cite the accompanying manuscript (details to be added upon publication).
