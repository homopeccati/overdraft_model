"""
Hyperparameter Tuning Module
============================

Uses Optuna to optimise model hyperparameters via StratifiedKFold CV.

Primary target: LightGBM (priority — fast + quality on tabular data).
Key parameters tuned:
  - n_estimators, learning_rate, max_depth, num_leaves
  - min_child_samples, subsample, colsample_bytree
  - reg_alpha, reg_lambda

Usage
-----
    from src.model.tuning import tune_hyperparams

    best_params = tune_hyperparams(X_train, y_train, n_trials=50)
    # best_params["params"] -> dict of best hyperparameters
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
N_FOLDS = 3          # fewer folds during tuning for speed

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

_optuna_available = False
try:
    import optuna
    _optuna_available = True
except ImportError:
    optuna = None


# ===================================================================
# LightGBM objective
# ===================================================================

def _make_lgb_objective(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
):
    """Return an Optuna objective function for LightGBM tuning."""
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }

        # Early stopping based on n_estimators
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 1000, step=50)

        oof_preds = np.empty(len(y))
        oof_preds[:] = np.nan
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.callback.early_stopping(10), lgb.callback.log_evaluation(0)],
            )
            y_proba = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = y_proba
            scores.append(roc_auc_score(y_val, y_proba))

        mean_auc = np.mean(scores)
        return mean_auc

    return objective


# ===================================================================
# XGBoost objective
# ===================================================================

def _make_xgb_objective(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
):
    """Return an Optuna objective function for XGBoost tuning."""
    import xgboost as xgb

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    imbalance_ratio = (len(y) - y.sum()) / y.sum()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": imbalance_ratio,
            "random_state": random_state,
            "verbosity": 0,
            "n_jobs": -1,
        }

        oof_preds = np.empty(len(y))
        oof_preds[:] = np.nan
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            y_proba = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = y_proba
            scores.append(roc_auc_score(y_val, y_proba))

        return np.mean(scores)

    return objective


# ===================================================================
# CatBoost objective
# ===================================================================

def _make_cb_objective(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
):
    """Return an Optuna objective function for CatBoost tuning."""
    import catboost as cb

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "auto_class_weights": "Balanced",
            "random_seed": random_state,
            "verbose": 0,
            "allow_writing_files": False,
        }

        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = cb.CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=20)
            y_proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_proba))

        return np.mean(scores)

    return objective


# ===================================================================
# Tuning runner
# ===================================================================

def tune_hyperparams(
    X: pd.DataFrame,
    y: pd.Series = None,
    target_col: str = "target_value",
    model_name: str = "LightGBM",
    n_trials: int = 30,
    n_folds: int = N_FOLDS,
    timeout: int = None,
    direction: str = "maximize",
    storage: str = None,
    study_name: str = None,
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run Optuna hyperparameter optimisation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix **or** full training DataFrame (if y is None).
    y : pd.Series or None
        Target labels.
    target_col : str
        Column name for target (used when y is None).
    model_name : str
        Which model to tune: "LightGBM", "XGBoost", or "CatBoost".
    n_trials : int
        Number of Optuna trials.
    n_folds : int
        Number of CV folds per trial.
    timeout : int or None
        Time limit in seconds (None = no limit).
    direction : str
        "maximize" or "minimize".
    storage : str or None
        Optuna storage URL for resuming (e.g. "sqlite:///study.db").
    study_name : str or None
        Name for the Optuna study.
    plot : bool
        If True, display optimisation history plots.
    verbose : bool
        If True, print progress.

    Returns
    -------
    dict with keys:
        params        — best hyperparameters
        best_score    — best CV ROC-AUC
        study         — Optuna study object
        model_name    — name of the tuned model
        n_trials      — number of completed trials
    """
    if not _optuna_available:
        raise ImportError(
            "Optuna is required. Install with: pip install optuna"
        )

    print("=" * 60)
    print(f"HYPERPARAMETER TUNING — {model_name}")
    print("=" * 60)

    # Separate target if needed
    if y is None:
        if target_col in X.columns:
            y = X[target_col].copy()
            X_feat = X.drop(columns=[target_col])
        else:
            raise ValueError(f"y is None and '{target_col}' not found in X.")
    else:
        X_feat = X.copy()

    # Keep only numeric
    X_num = X_feat.select_dtypes(include=[np.number])
    dropped = set(X_feat.columns) - set(X_num.columns)
    if dropped and verbose:
        print(f"  Dropped non-numeric: {dropped}")

    if verbose:
        print(f"\n  Model:      {model_name}")
        print(f"  Trials:     {n_trials}")
        print(f"  CV folds:   {n_folds}")
        print(f"  Features:   {X_num.shape[1]}")
        print(f"  Samples:    {len(y):,}")

    # Select objective function
    if model_name == "LightGBM":
        objective_fn = _make_lgb_objective(X_num, y, n_folds=n_folds)
    elif model_name == "XGBoost":
        objective_fn = _make_xgb_objective(X_num, y, n_folds=n_folds)
    elif model_name == "CatBoost":
        objective_fn = _make_cb_objective(X_num, y, n_folds=n_folds)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose LightGBM, XGBoost, or CatBoost.")

    # Create / load study
    if storage is not None:
        if study_name is None:
            study_name = f"{model_name.lower()}_tuning"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction=direction,
        )
    else:
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
        )

    # Progress bar
    if verbose:
        print(f"\n  Running {n_trials} trials...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)

    # Results
    best_params = study.best_params
    best_score = study.best_value
    best_trial = study.best_trial

    if verbose:
        print(f"\n  Best trial:   #{best_trial.number}")
        print(f"  Best ROC-AUC: {best_score:.4f}")
        print(f"\n  Best params:")
        for k, v in best_params.items():
            print(f"    {k:25s} = {v}")

        # Parameter importances
        try:
            importances = optuna.importance.get_param_importances(study)
            print(f"\n  Parameter importances:")
            for k, v in sorted(importances.items(), key=lambda x: -x[1]):
                print(f"    {k:25s} = {v:.3f}")
        except Exception:
            pass

    # Plots
    if plot:
        _plot_tuning_results(study, model_name, best_params)

    return {
        "params": best_params,
        "best_score": best_score,
        "study": study,
        "model_name": model_name,
        "n_trials": len(study.trials),
    }


# ===================================================================
# Plotting
# ===================================================================

def _plot_tuning_results(study, model_name: str, best_params: dict):
    """Plot optimisation history and parameter importances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Optimisation history
    ax = axes[0]
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    ax.plot(range(1, len(values) + 1), values, "o-", color="steelblue", markersize=3, linewidth=1)
    ax.axhline(study.best_value, color="red", linestyle="--", alpha=0.6,
               label=f"Best: {study.best_value:.4f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Optimisation History — {model_name}")
    ax.legend()

    # 2. Parameter importances
    ax = axes[1]
    try:
        importances = optuna.importance.get_param_importances(study)
        params_sorted = sorted(importances.items(), key=lambda x: -x[1])
        names = [p[0] for p in params_sorted]
        scores = [p[1] for p in params_sorted]
        colors = sns.color_palette("viridis", len(names))
        ax.barh(range(len(names)), scores, color=colors, edgecolor="gray")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importance (Optuna)")
    except Exception:
        ax.text(0.5, 0.5, "Parameter importance\nnot available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()


# ===================================================================
# Standalone entry point
# ===================================================================
if __name__ == "__main__":
    import os
    from src.model.preprocessing import preprocess_data
    from src.model.features import create_features

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train_apps.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_apps.csv"))

    X_tr, X_te, _ = preprocess_data(train_df, test_df)
    X_tr, X_te, _ = create_features(X_tr, X_te, verbose=False)
    y_tr = train_df["target_value"]

    result = tune_hyperparams(X_tr, y_tr, model_name="LightGBM", n_trials=10, plot=False)
    print(f"\nTuning complete. Best ROC-AUC: {result['best_score']:.4f}")
    print(f"Best params: {result['params']}")
