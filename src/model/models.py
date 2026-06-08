"""
Main Models Module
==================

Trains and compares gradient boosting models with StratifiedKFold CV:
  - LightGBM  (priority — fast + quality on tabular data)
  - XGBoost   (alternative)
  - CatBoost  (native categorical support — though cats already encoded)
  - Random Forest (benchmark)

All models use OOF (out-of-fold) predictions for fair comparison.
The primary metric is ROC-AUC; secondary: Gini, F1, Precision, Recall.

Usage
-----
    from src.model.models import train_models

    results = train_models(X_train, y_train)
    # or:   results = train_models(train_df)  # if DataFrame has target_value
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
N_FOLDS = 5

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------
# Optional imports with graceful fallbacks
# ---------------------------------------------------------------------------
_lightgbm_available = False
_xgboost_available = False
_catboost_available = False

try:
    import lightgbm as lgb
    _lightgbm_available = True
except ImportError:
    lgb = None

try:
    import xgboost as xgb
    _xgboost_available = True
except ImportError:
    xgb = None

try:
    import catboost as cb
    _catboost_available = True
except ImportError:
    cb = None


# ===================================================================
# Model configurations
# ===================================================================

def _get_model_configs(random_state: int = RANDOM_STATE) -> list:
    """Return list of (name, estimator, available_flag) tuples."""
    configs = []

    # LightGBM
    if _lightgbm_available:
        configs.append((
            "LightGBM",
            lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight="balanced",
                random_state=random_state,
                verbosity=-1,
                n_jobs=-1,
            ),
            True,
        ))

    # XGBoost
    if _xgboost_available:
        configs.append((
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=15.42,  # imbalance ratio
                random_state=random_state,
                verbosity=0,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
            True,
        ))

    # CatBoost
    if _catboost_available:
        configs.append((
            "CatBoost",
            cb.CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                random_seed=random_state,
                verbose=0,
                allow_writing_files=False,
            ),
            True,
        ))

    # Random Forest (always available via sklearn)
    configs.append((
        "RandomForest",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        ),
        True,
    ))

    return configs


# ===================================================================
# Single-model CV trainer
# ===================================================================

def _train_model_cv(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Run StratifiedKFold CV for a single model.

    Returns dict with: cv_scores, oof_preds, mean_metrics, std_metrics,
                       feature_importance (if available), model (full-data fit).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = []
    oof_preds = np.empty(len(y))
    oof_preds[:] = np.nan

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = deepcopy(model)
        fold_model.fit(X_tr, y_tr)

        if hasattr(fold_model, "predict_proba"):
            y_proba = fold_model.predict_proba(X_val)[:, 1]
        else:
            y_proba = fold_model.predict(X_val)

        # Threshold by F1
        thresholds = np.linspace(0.01, 0.99, 100)
        best_f1, best_t, best_pred = 0, 0.5, None
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
                best_pred = y_pred

        roc_auc = roc_auc_score(y_val, y_proba)
        gini = 2 * roc_auc - 1
        f1 = f1_score(y_val, best_pred, zero_division=0)
        prec = precision_score(y_val, best_pred, zero_division=0)
        rec = recall_score(y_val, best_pred, zero_division=0)

        cv_scores.append({
            "fold": fold,
            "roc_auc": roc_auc,
            "gini": gini,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "threshold": best_t,
        })
        oof_preds[val_idx] = y_proba

    # Mean metrics
    scores_df = pd.DataFrame(cv_scores)
    mean_m = scores_df.drop(columns=["fold"]).mean()
    std_m = scores_df.drop(columns=["fold"]).std()

    # Full-data fit
    final_model = deepcopy(model)
    final_model.fit(X, y)

    # Feature importance
    imp = _get_feature_importance(final_model, X.columns)

    return {
        "name": name,
        "cv_scores": cv_scores,
        "mean_metrics": mean_m.to_dict(),
        "std_metrics": std_m.to_dict(),
        "oof_preds": oof_preds,
        "oof_labels": y.values,
        "feature_importance": imp,
        "model": final_model,
    }


def _get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from a fitted model, if available."""
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "importance": fi,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    elif hasattr(model, "coef_"):
        coef = model.coef_[0]
        return pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(coef),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})


# ===================================================================
# Main entry point
# ===================================================================

def train_models(
    X: pd.DataFrame,
    y: pd.Series = None,
    target_col: str = "target_value",
    n_splits: int = N_FOLDS,
    models: list = None,
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Train and compare multiple classifiers with StratifiedKFold CV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix **or** full training DataFrame (if ``y`` is None,
        ``target_col`` is extracted from X).
    y : pd.Series or None
        Target labels.  If None, extracted from ``X[target_col]``.
    target_col : str
        Column name for the target (used only when ``y`` is None).
    n_splits : int
        Number of CV folds.
    models : list of (name, estimator) or None
        Custom model list.  If None, uses all available defaults.
    plot : bool
        If True, display comparison plots.
    verbose : bool
        If True, print detailed results.

    Returns
    -------
    dict with keys:
        comparison_df  — DataFrame comparing all models
        results        — dict mapping model name → per-model result dict
        best_model     — name of the best model (by mean ROC-AUC)
    """
    print("=" * 60)
    print("MAIN MODELS — Comparison")
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
    X_feat_num = X_feat.select_dtypes(include=[np.number])
    dropped = set(X_feat.columns) - set(X_feat_num.columns)
    if dropped and verbose:
        print(f"  Dropped non-numeric: {dropped}")

    # Determine which models to run
    if models is not None:
        model_configs = models
    else:
        model_configs = _get_model_configs()
        # Warn about missing libraries
        if not _lightgbm_available and verbose:
            print("  LightGBM not available — skipping.  Install with: pip install lightgbm")
        if not _xgboost_available and verbose:
            print("  XGBoost not available — skipping.   Install with: pip install xgboost")
        if not _catboost_available and verbose:
            print("  CatBoost not available — skipping.  Install with: pip install catboost")

    n_positive = y.sum()
    n_negative = len(y) - n_positive
    imbalance_ratio = n_negative / n_positive if n_positive > 0 else float("inf")

    if verbose:
        print(f"\n  Samples:   {len(y):,}")
        print(f"  Positive:  {n_positive:,} ({100 * n_positive / len(y):.2f}%)")
        print(f"  Features:  {X_feat_num.shape[1]}")
        print(f"  CV folds:  {n_splits}")

    # ── Train each model ────────────────────────────────────────────
    all_results = {}

    for name, estimator, available in model_configs:
        if not available:
            continue

        if verbose:
            print(f"\n  >>> {name}")

        result = _train_model_cv(
            name, estimator, X_feat_num, y,
            n_splits=n_splits, random_state=RANDOM_STATE,
        )
        all_results[name] = result

        if verbose:
            m = result["mean_metrics"]
            s = result["std_metrics"]
            print(f"      ROC-AUC: {m['roc_auc']:.4f} ± {s['roc_auc']:.4f}"
                  f"   Gini: {m['gini']:.4f}"
                  f"   F1: {m['f1']:.4f}"
                  f"   Prec: {m['precision']:.4f}"
                  f"   Recall: {m['recall']:.4f}")

    # ── Comparison table ────────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        m = res["mean_metrics"]
        rows.append({
            "Model": name,
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "Gini": f"{m['gini']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
        })

    comparison_df = pd.DataFrame(rows)
    if verbose:
        print(f"\n  {'─' * 60}")
        print("  COMPARISON TABLE")
        print(f"  {'─' * 60}")
        print(comparison_df.to_string(index=False))
        print(f"  {'─' * 60}")

    # Best by ROC-AUC
    best_name = max(all_results, key=lambda n: all_results[n]["mean_metrics"]["roc_auc"])
    if verbose:
        print(f"\n  Best model: {best_name}"
              f"  (ROC-AUC = {all_results[best_name]['mean_metrics']['roc_auc']:.4f})")

    # ── Plots ───────────────────────────────────────────────────────
    if plot:
        _plot_model_comparison(all_results, comparison_df)

    return {
        "comparison_df": comparison_df,
        "results": all_results,
        "best_model": best_name,
    }


# ===================================================================
# Plotting
# ===================================================================

def _plot_model_comparison(all_results: dict, comparison_df: pd.DataFrame):
    """Produce comparison plots for all trained models."""
    n_models = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. ROC curves (OOF)
    ax = axes[0, 0]
    colors = sns.color_palette("Set2", n_models)
    for (name, res), color in zip(all_results.items(), colors):
        fpr, tpr, _ = roc_curve(res["oof_labels"], res["oof_preds"])
        auc = roc_auc_score(res["oof_labels"], res["oof_preds"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (OOF Predictions)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # 2. Metric comparison bar chart
    ax = axes[0, 1]
    metrics_df = comparison_df.copy()
    for col in ["ROC-AUC", "Gini", "F1", "Precision", "Recall"]:
        metrics_df[col] = metrics_df[col].astype(float)
    plot_df = metrics_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", ax=ax,
                palette="Set2", edgecolor="gray")
    ax.set_title("Model Metrics Comparison")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1)

    # 3. OOF probability distributions
    ax = axes[1, 0]
    for (name, res), color in zip(all_results.items(), colors):
        sns.kdeplot(res["oof_preds"], label=name, color=color, ax=ax, linewidth=1.5)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("OOF Probability Distributions")
    ax.legend(fontsize=8)

    # 4. Top features from best model
    ax = axes[1, 1]
    best_name = max(all_results, key=lambda n: all_results[n]["mean_metrics"]["roc_auc"])
    best_imp = all_results[best_name]["feature_importance"]
    if best_imp is not None and "importance" in best_imp.columns:
        top15 = best_imp.head(15)
        ax.barh(range(len(top15)), top15["importance"], color="steelblue", edgecolor="gray")
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 15 Features ({best_name})")

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

    results = train_models(X_tr, y_tr, plot=False)
    print(f"\nBest model: {results['best_model']}")
