# Project Guide: Overdraft Offer Acceptance Prediction Model

## Project Overview

This project builds a **binary classification model** to predict whether a small/medium business client will accept a bank's overdraft offer. The target variable `target_value` is 1 (accepted) or 0 (declined).

### Key Technologies

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Primary language |
| **pandas, numpy** | Data manipulation |
| **matplotlib, seaborn** | Visualization |
| **scikit-learn** | Preprocessing, metrics, baseline models |
| **LightGBM** | Primary gradient boosting model (speed + quality) |
| **XGBoost** | Gradient boosting alternative |
| **CatBoost** | Gradient boosting with native categorical support |
| **shap** | Model interpretability |
| **optuna** | Hyperparameter optimization |
| **joblib** | Model persistence |

### High-Level Architecture

```
Raw Data (CSV) → src/model/eda.py → src/model/preprocessing.py → src/model/baseline.py →
src/model/models.py → src/model/feature_selection.py → src/model/tuning.py → src/model/final_model.py
                                                                           ↓
                                                                 pipeline.ipynb (orchestration + display)
```

All logic resides in `src/model/` modules. The notebook imports them, passes data between steps, and displays results (plots, tables, metrics).

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for version control)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd <project-directory>

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost shap optuna joblib jupyter

# Or use requirements.txt after creating it:
# pip install -r requirements.txt
```

### Basic Usage

```python
import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('data/train_apps.csv')
test = pd.read_csv('data/test_apps.csv')

# Quick sanity check
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target distribution:\n{train['target_value'].value_counts(normalize=True)}")
```

### Running Tests

There are no formal test suites yet. Testing is done manually via:
- Running individual `src/model/` modules and inspecting console output
- Cross-validation metrics during model development
- Visual inspection of distributions and SHAP plots from the Jupyter notebook

> **Note:** Add unit tests for preprocessing functions (`src/model/preprocessing.py`) and feature engineering (`src/model/features.py`) as the project matures.

## Project Structure

```
mfti/
├── data/                           # Dataset files (not tracked in git ideally)
│   ├── train_apps.csv              # Training data (~45 MB)
│   └── test_apps.csv               # Test data (~12 MB, no target)
│
├── src/
│   ├── model/                      # All logic — reusable Python modules, importable
│   │   ├── eda.py                  # Exploratory Data Analysis functions
│   │   ├── preprocessing.py        # Data cleaning, transformers, pipelines
│   │   ├── features.py             # Feature engineering functions
│   │   ├── baseline.py             # Logistic Regression baseline
│   │   ├── models.py               # Model training (LightGBM, XGBoost, CatBoost, RF)
│   │   ├── feature_selection.py    # SHAP analysis, feature importance
│   │   ├── tuning.py               # Optuna hyperparameter optimization
│   │   ├── final_model.py          # Final model training & prediction generation
│   │   └── evaluation.py           # Metrics & visualization helpers
│   └── mfti.egg-info/              # Package metadata (auto-generated)
│
├── notebooks/                      # Single Jupyter notebook for orchestration
│   └── pipeline.ipynb              # Orchestrates all src/model/ modules, displays results
│
├── models/                         # Saved trained models
│   └── final_model.pkl             # Trained model (joblib dump)
│
├── outputs/                        # Generated outputs
│   └── predictions.csv             # Predictions for test set
│
├── .continue/
│   ├── rules/                      # Agent rules & project guide
│   │   ├── AGENTS.md               # This file — project guide for Continue
│   │   ├── ar1_surgical_changes.md
│   │   ├── ar2_deletion_default.md
│   │   ├── ar3_no_speculative_additions.md
│   │   ├── ar4_output_diff.md
│   │   ├── ar5_justify_abstraction.md
│   │   ├── ar6_measure_length_reduction.md
│   │   ├── ar7_preserve_public_contracts.md
│   │   ├── ar8_plain_language_audit.md
│   │   └── ar9_clarify_before_you_code.md
│   └── prompts/                    # One-shot prompts (p1–p10)
│       ├── p1_compression_pass.md
│       ├── p2_dead_code_finder.md
│       ├── p3_abstraction_audit.md
│       ├── p4_targeted_refactor.md
│       ├── p5_comment_purge.md
│       ├── p6_readme_reconstruction.md
│       ├── p7_docstring_triage.md
│       ├── p8_architecture_doc_triage.md
│       ├── p9_pre_refactor_audit.md
│       └── p10_regression_safety_check.md
│
├── overdraft model copilot instructions.md  # Detailed step-by-step plan (Russian)
├── Описание переменных.md                   # Variable descriptions (Russian)
├── pyproject.toml                  # Project configuration (uv)
├── README.md                       # Project readme
└── .venv/                          # Virtual environment (not tracked)
```

### Key Files

| File | Role |
|---|---|
| `src/model/*.py` | All logic (EDA, preprocessing, features, models, tuning, evaluation) — importable |
| `notebooks/pipeline.ipynb` | Single Jupyter notebook that orchestrates the pipeline |
| `data/train_apps.csv` | Training dataset with target variable |
| `data/test_apps.csv` | Test dataset (no target — used for submission) |
| `overdraft model copilot instructions.md` | Complete project roadmap with methodology, code style, and pitfalls |
| `Описание переменных.md` | Detailed descriptions of all features (Russian) |

### Variables Overview

| Variable | Description |
|---|---|
| `front_id` | Unique application ID |
| `decision_day` | Decision date |
| `loan_amount_last` | Requested loan amount |
| `overdraft_limit_min` | Minimum overdraft limit (pre-scoring) |
| `overdraft_limit_max` | Maximum overdraft limit (pre-scoring) |
| `offered_rate` | Offered interest rate |
| `cb_rate` | Central Bank key rate at application time |
| `target_value` | **Target**: 1 = accepted, 0 = declined |

See `Описание переменных.md` for the complete list with detailed descriptions.

## Development Workflow

### Coding Standards

- **Python 3.10+** with **PEP 8** style
- Use descriptive variable names (English)
- Comment non-trivial decisions — prefer explaining *why* over *what*
- Break code into logical sections with header comments
- Fix `random_state=42` everywhere for reproducibility

### Module Contract (src/model/ modules → notebook)

Every `src/model/` module must be importable and callable from a Jupyter notebook:

```python
from src.model.eda import run_eda
from src.model.preprocessing import preprocess_data
from src.model.features import create_features
from src.model.baseline import train_baseline
from src.model.models import train_models
from src.model.feature_selection import analyze_features
from src.model.tuning import tune_hyperparams
from src.model.final_model import train_final_model, generate_predictions

train, test = run_eda(train_path, test_path)
train, test = preprocess_data(train, test)
train, test = create_features(train, test)
baseline_results = train_baseline(train)
model_results = train_models(train)
selected_features = analyze_features(model_results, train)
best_params = tune_hyperparams(train, selected_features)
predictions = train_final_model(train, test, best_params)
```

Each function returns data (DataFrames, dicts, plots) that the notebook can display.

### Testing Approach

- **No leaks**: All transformations fit on train only, applied to test via `Pipeline` or manual alignment
- **Stratified Cross-Validation**: `StratifiedKFold(n_splits=5)` due to potential class imbalance
- **Primary metric**: ROC-AUC (robust to imbalanced data)
- **Secondary metrics**: Gini, F1, Precision, Recall

### Recommended Library Stack

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import shap
import optuna
import joblib
```

### Build and Deployment

Currently no build/deployment pipeline. The model is developed locally with development logic in `src/model/` Python modules and orchestration via a single Jupyter notebook. The final artifact is a `.pkl` file. Consider adding:
- `Dockerfile` for containerized inference
- `requirements.txt` for reproducible environments
- CI/CD pipeline (GitHub Actions) for automated testing

### Contribution Guidelines

1. Create a feature branch from `main`
2. Keep all development logic in `src/model/` Python modules — each module is a self-contained step
3. Use `notebooks/pipeline.ipynb` only for orchestration and display
4. Document key decisions and findings
5. Submit a PR with a summary of changes and results

## Key Concepts

### Domain-Specific Terminology

- **Overdraft**: A credit line where a client can withdraw more than their account balance
- **Offer**: The bank's proposal including limit range and interest rate
- **Pre-scoring**: Initial creditworthiness assessment determining min/max limits
- **BKI (Bureau of Credit Histories)**: External credit history data source
- **OOF (Out-Of-Fold) predictions**: Predictions from cross-validation where each fold's model predicts on unseen data

### Core Abstractions

- **Target Encoding**: Encoding categorical features by their target mean (with smoothing for rare categories)
- **Out-of-Fold Encoding**: Encoding categorical variables using fold-specific statistics to prevent data leakage
- **Feature Engineering Ideas** (from the project plan):

| Engineered Feature | Formula | Rationale |
|---|---|---|
| `rate_spread` | `offered_rate - cb_rate` | Offer attractiveness relative to market |
| `limit_request_ratio` | `loan_amount_last / overdraft_limit_max` | How well request matches offer |
| `limit_range` | `overdraft_limit_max - overdraft_limit_min` | Width of offer corridor |
| `activity_ratio_30_90` | `cnt_deb_ul_ip_30 / (cnt_deb_ul_ip_90 + 1)` | Recent activity dynamics |
| `log_balance` | `log1p(balance_rur_amt_30_min)` | Log-transformed balance |

### Design Patterns

- **Module-per-step pattern**: Each pipeline stage is a separate importable `.py` file
- **Data flows as return values**: Modules return DataFrames/metrics; notebook chains them
- **Pipeline pattern**: Chain transformations using `sklearn.pipeline.Pipeline`
- **Stratified K-Fold**: Preserve class distribution across CV folds
- **Gradient Boosting ensemble**: Sequential tree-based learners
- **SHAP explainability**: Game-theoretic feature importance attribution

## Troubleshooting

### Common Issues

| Problem | Likely Cause | Solution |
|---|---|---|
| `balance_rur_amt_30_min` has negative values | Overdraft usage — account went negative | Don't log-transform; clip or use raw values |
| Right-skewed transaction features | `sum_deb_*`, `cnt_deb_*` have long tails | Apply `np.log1p()` transformation |
| Rare categories in `fl_adminarea` | Some regions have < 1% frequency | Group into `other` category |
| Target Encoding overfitting on small categories | Insufficient samples per category | Use out-of-fold encoding with smoothing |
| `decision_day` as feature | Risk of temporal leakage | Only use engineered temporal features (lags, trends) |
| Data leakage from preprocessing | Fitting transformers on full dataset | Always `fit` on train, `transform` on test |
| Class imbalance | Target may have skewed distribution | Use ROC-AUC; try `scale_pos_weight` or SMOTE |

### Debugging Tips

1. **Always check shapes**: After each transformation, verify `train.shape` and that features align between train/test
2. **Run modules standalone**: Each `src/model/` module can be run as `python -c "from src.model.eda import run_eda; ..."` to test in isolation
3. **Monitor OOF vs train scores**: Large gap = overfitting
4. **Visualize predictions**: Plot probability distributions for class 0 vs class 1
5. **Check for duplicates**: `train['front_id'].duplicated().sum()` — there shouldn't be any

### Known Pitfalls

- `balance_rur_amt_30_min` may contain negative values (overdraft) — handle carefully, avoid `log` on negatives
- Transaction variables (`sum_deb_*`, `cnt_deb_*`) are typically right-skewed — recommend `log1p` transformation
- `decision_day` should not be used as a direct feature without careful thought (risk of leakage or pseudo-temporal trend)
- `fl_adminarea` may contain rare regions — group into `other` at frequency < 1%
- For Target Encoding of `db_group_last`, use out-of-fold encoding to prevent data leakage

## References

### Documentation

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Project Documents

- `overdraft model copilot instructions.md` — Complete methodology, code style rules, and step-by-step plan
- `Описание переменных.md` — Variable descriptions in Russian
- `anti-neuroslop-toolkit.md` — Original source document for agent rules and prompts
- `.continue/rules/ar1.md` through `ar9.md` — Standalone rule files extracted from the toolkit
- `.continue/rules/prompts/p1.md` through `p10.md` — Standalone prompt files extracted from the toolkit

### Methodology References

- [Gradient Boosting overview](https://explained.ai/gradient-boosting/)
- [SHAP for model interpretability](https://christophm.github.io/interpretable-ml-book/shap.html)
- [Target Encoding with smoothing](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
- [Stratified K-Fold Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

---

## Updating This Guide

As the project evolves, keep this file updated:
- Add new feature engineering approaches as they are discovered
- Document model performance benchmarks
- Add troubleshooting entries for issues encountered
- Update installation instructions if dependencies change

> **Tip for Continue users**: This file is automatically loaded into context when working with this project. Create additional `*.md` files in subdirectories (e.g., `src/model/rules.md`) for component-specific documentation.
