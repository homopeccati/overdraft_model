"""
Final Model Module
==================

Trains the final production model with optimised hyperparameters,
calibrates probabilities, generates SHAP interpretation, and
produces test-set predictions.

Pipeline
--------
  1. Train final model on full training data (best params)
  2. Calibrate probabilities (Platt / Isotonic)
  3. SHAP global interpretation
  4. Save model to ``models/final_model.pkl``
  5. Generate predictions on test set
  6. Save ``outputs/predictions.csv``

Usage
-----
    from src.model.final_model import train_final_model, generate_predictions

    result = train_final_model(X_train, y_train, best_params)
    predictions = generate_predictions(result["model"], result["calibrator"], X_test)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from copy import deepcopy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# Paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

_shap_available = False
try:
    import shap
    _shap_available = True
except ImportError:
    shap = None


# ===================================================================
# 1. Train final model
# ===================================================================

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series = None,
    target_col: str = "target_value",
    best_params: dict = None,
    model_name: str = "LightGBM",
    calibrate: bool = True,
    calibration_method: str = "isotonic",
    calibration_cv: int = 5,
    save_model: bool = True,
    model_path: str = None,
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Train the final production model with optimised hyperparameters.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix **or** full training DataFrame (if y is None).
    y : pd.Series or None
        Target labels.
    target_col : str
        Column name for the target (used when y is None).
    best_params : dict or None
        Best hyperparameters from tuning.  If None, uses defaults.
    model_name : str
        Which model: "LightGBM" (default), "XGBoost", "CatBoost", "RandomForest".
    calibrate : bool
        If True, apply ``CalibratedClassifierCV`` (Platt / Isotonic).
    calibration_method : str
        "sigmoid" (Platt) or "isotonic".
    calibration_cv : int
        Number of folds for calibration (``cv`` parameter).
    save_model : bool
        If True, save model to disk.
    model_path : str or None
        Path for saved model.  If None, uses ``models/final_model.pkl``.
    plot : bool
        If True, display diagnostic plots.
    verbose : bool
        If True, print progress.

    Returns
    -------
    dict with keys:
        model        — trained model (before calibration)
        calibrator   — ``CalibratedClassifierCV`` or None
        metrics      — dict of train-set metrics
        shap_result  — SHAP analysis result (or None)
        model_name   — name of the trained model
        best_params  — hyperparameters used
    """
    print("=" * 60)
    print("FINAL MODEL")
    print("=" * 60)

    # Separate target
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

    # Default params if none provided
    if best_params is None:
        best_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 7,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }

    n_pos = y.sum()
    n_neg = len(y) - n_pos

    if verbose:
        print(f"\n  Model:          {model_name}")
        print(f"  Samples:        {len(y):,}")
        print(f"  Positive:       {n_pos:,} ({100 * n_pos / len(y):.2f}%)")
        print(f"  Features:       {X_num.shape[1]}")
        print(f"  Calibration:    {calibration_method if calibrate else 'None'}")
        print(f"\n  Params:")
        for k, v in best_params.items():
            print(f"    {k:25s} = {v}")

    # ── Train base model ────────────────────────────────────────────
    if verbose:
        print(f"\n  Training {model_name} on full data...")

    if model_name == "LightGBM":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            **best_params,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        import xgboost as xgb
        imbalance = (len(y) - y.sum()) / y.sum()
        model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=imbalance,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )
    elif model_name == "CatBoost":
        import catboost as cb
        model = cb.CatBoostClassifier(
            **best_params,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )
    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            **{k: v for k, v in best_params.items()
               if k in ("n_estimators", "max_depth", "min_samples_leaf",
                        "min_samples_split", "max_features")},
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_num, y)
    y_proba = model.predict_proba(X_num)[:, 1]

    # ── Calibration ─────────────────────────────────────────────────
    calibrator = None
    if calibrate:
        if verbose:
            print(f"  Calibrating ({calibration_method}, {calibration_cv}-fold)...")
        calibrator = CalibratedClassifierCV(
            deepcopy(model),
            method=calibration_method,
            cv=calibration_cv,
        )
        calibrator.fit(X_num, y)
        y_calibrated = calibrator.predict_proba(X_num)[:, 1]
    else:
        y_calibrated = y_proba

    # ── Metrics ─────────────────────────────────────────────────────
    # Threshold by F1
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        f1 = f1_score(y, (y_calibrated >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    y_pred = (y_calibrated >= best_t).astype(int)
    roc_auc = roc_auc_score(y, y_calibrated)
    f1 = f1_score(y, y_pred, zero_division=0)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)

    metrics = {
        "roc_auc": roc_auc,
        "gini": 2 * roc_auc - 1,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "threshold": best_t,
    }

    if verbose:
        print(f"\n  ── Metrics (train) ──")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Gini:      {metrics['gini']:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  Threshold: {best_t:.4f}")

    # ── SHAP ────────────────────────────────────────────────────────
    shap_result = None
    if _shap_available:
        if verbose:
            print(f"\n  Computing SHAP values...")
        shap_result = _compute_shap(model, X_num)

        if plot and shap_result is not None:
            _plot_shap(shap_result)
            _plot_waterfall(shap_result)

    # ── Calibration curve ───────────────────────────────────────────
    if plot and calibrate:
        _plot_calibration(y, y_proba, y_calibrated)

    # ── Save model ──────────────────────────────────────────────────
    if save_model:
        if model_path is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, "final_model.pkl")

        save_dict = {
            "model": model,
            "calibrator": calibrator,
            "best_params": best_params,
            "model_name": model_name,
            "metrics": metrics,
            "features": list(X_num.columns),
        }
        joblib.dump(save_dict, model_path)
        if verbose:
            print(f"\n  Model saved: {model_path}")

    return {
        "model": model,
        "calibrator": calibrator,
        "metrics": metrics,
        "shap_result": shap_result,
        "model_name": model_name,
        "best_params": best_params,
    }


# ===================================================================
# 2. Generate predictions
# ===================================================================

def generate_predictions(
    model,
    X_test: pd.DataFrame,
    front_ids: pd.Series = None,
    calibrator=None,
    threshold: float = 0.5,
    save: bool = True,
    output_path: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate predictions on test data and optionally save to CSV.

    Parameters
    ----------
    model : estimator
        Trained model (predict_proba-capable).
    X_test : pd.DataFrame
        Test features (preprocessed + engineered).
    front_ids : pd.Series or None
        Application IDs.  If None, uses X_test index.
    calibrator : ``CalibratedClassifierCV`` or None
        Optional fitted calibrator.
    threshold : float
        Classification threshold for predicted labels.
    save : bool
        If True, save to ``outputs/predictions.csv``.
    output_path : str or None
        Custom output path.
    verbose : bool
        If True, print summary.

    Returns
    -------
    pd.DataFrame with columns: front_id, predicted_proba, [predicted_label]
    """
    X_num = X_test.select_dtypes(include=[np.number])

    if calibrator is not None:
        y_proba = calibrator.predict_proba(X_num)[:, 1]
    else:
        y_proba = model.predict_proba(X_num)[:, 1]

    y_pred = (y_proba >= threshold).astype(int)

    if front_ids is None:
        front_ids = np.arange(len(y_proba))

    pred_df = pd.DataFrame({
        "front_id": front_ids,
        "predicted_proba": y_proba,
        "predicted_label": y_pred,
    })

    if verbose:
        n_accepted = y_pred.sum()
        print(f"\n  Predictions: {len(pred_df):,} samples")
        print(f"  Accepted:    {n_accepted:,} ({100 * n_accepted / len(pred_df):.2f}%)")
        print(f"  Declined:    {len(pred_df) - n_accepted:,} ({100 * (len(pred_df) - n_accepted) / len(pred_df):.2f}%)")
        print(f"  Threshold:   {threshold:.4f}")

    if save:
        if output_path is None:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
        pred_df.to_csv(output_path, index=False)
        if verbose:
            print(f"  Predictions saved: {output_path}")

    return pred_df


# ===================================================================
# 3. SHAP helpers
# ===================================================================

def _compute_shap(model, X: pd.DataFrame, sample_size: int = 500):
    """Compute SHAP values on a sample."""
    if len(X) > sample_size:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature": X_sample.columns,
            "shap_importance": mean_abs_shap,
        }).sort_values("shap_importance", ascending=False)

        return {
            "shap_values": shap_values,
            "importance_df": imp_df,
            "X_sample": X_sample,
            "explainer": explainer,
        }
    except Exception as e:
        print(f"  SHAP skipped: {e}")
        return None


def _plot_shap(shap_result: dict, max_features: int = 20):
    """SHAP summary (beeswarm) + bar."""
    if shap_result is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * max_features)))

    ax = axes[0]
    top = shap_result["importance_df"].head(max_features)
    colors = sns.color_palette("viridis", len(top))
    ax.barh(range(len(top)), top["shap_importance"], color=colors, edgecolor="gray")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("SHAP Feature Importance (Final Model)")

    ax = axes[1]
    shap.summary_plot(
        shap_result["shap_values"].values[:, :max_features],
        shap_result["X_sample"].iloc[:, :max_features],
        plot_type="dot", show=False, max_display=max_features, alpha=0.6,
    )
    ax.set_title("SHAP Summary (beeswarm)")

    plt.tight_layout()
    plt.show()


def _plot_waterfall(shap_result: dict, index: int = 0):
    """Single-observation waterfall plot."""
    if shap_result is None or index >= len(shap_result["shap_values"]):
        return
    plt.figure(figsize=(8, 4))
    shap.waterfall_plot(shap_result["shap_values"][index], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Sample #{index}")
    plt.tight_layout()
    plt.show()


def _plot_calibration(y_true, y_raw, y_calibrated):
    """Calibration curve (reliability diagram)."""
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(7, 5))

    for probs, label, color in [(y_raw, "Raw", "steelblue"),
                                  (y_calibrated, "Calibrated", "crimson")]:
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=15)
        ax.plot(prob_pred, prob_true, "o-", label=label, color=color, linewidth=2)
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color=color)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ===================================================================
# Standalone entry point
# ===================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from src.model.preprocessing import preprocess_data
    from src.model.features import create_features
    from src.model.tuning import tune_hyperparams

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train_apps.csv"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test_apps.csv"))

    X_tr, X_te, _ = preprocess_data(train_df, test_df)
    X_tr, X_te, _ = create_features(X_tr, X_te, verbose=False)
    y_tr = train_df["target_value"]

    # Quick tuning
    tune_result = tune_hyperparams(X_tr, y_tr, model_name="LightGBM",
                                    n_trials=5, n_folds=2, plot=False)

    # Final model
    final = train_final_model(X_tr, y_tr, best_params=tune_result["params"],
                               model_name="LightGBM", plot=False)

    # Predictions
    preds = generate_predictions(
        final["model"], X_te,
        front_ids=test_df["front_id"],
        calibrator=final["calibrator"],
        threshold=final["metrics"]["threshold"],
    )
    print(f"\nDone. Predictions shape: {preds.shape}")
