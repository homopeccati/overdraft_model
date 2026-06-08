"""
Baseline Model Module
=====================

Trains a Logistic Regression baseline with StratifiedKFold cross-validation.

Provides:
  - `train_baseline()` — main entry point
  - `evaluate_threshold()` — optimal threshold search by F1
  - `plot_confusion_matrix()` — confusion matrix visualization

Usage
-----
    from src.model.baseline import train_baseline

    results = train_baseline(X_train, y_train)
    # or:   results = train_baseline(train)  # if train has target_value

    print(results["mean_metrics"])
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# ===================================================================
# Threshold optimization
# ===================================================================

def evaluate_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                       n_thresholds: int = 100) -> dict:
    """
    Find the optimal classification threshold by maximising F1 score.

    Returns a dict with keys: ``threshold``, ``f1``, ``precision``,
    ``recall``, and DataFrames ``curve`` (all thresholds) and
    ``optimal_row``.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    scores = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        scores.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    curve = pd.DataFrame(scores)
    best_idx = curve["f1"].idxmax()
    best = curve.loc[best_idx]

    return {
        "threshold": best["threshold"],
        "f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "curve": curve,
        "optimal_row": best,
    }


# ===================================================================
# Confusion matrix plot
# ===================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Confusion Matrix") -> plt.Figure:
    """Plot a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Declined (0)", "Accepted (1)"],
                yticklabels=["Declined (0)", "Accepted (1)"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ===================================================================
# Main baseline trainer
# ===================================================================

def train_baseline(
    X: pd.DataFrame,
    y: pd.Series = None,
    target_col: str = "target_value",
    n_splits: int = 5,
    scale_features: bool = True,
    max_iter: int = 2000,
    class_weight: str = "balanced",
    random_state: int = RANDOM_STATE,
    plot: bool = True,
) -> dict:
    """
    Train a Logistic Regression baseline with StratifiedKFold CV.

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
    scale_features : bool
        Whether to apply StandardScaler inside the pipeline.
    max_iter : int
        Solver max iterations.
    class_weight : str or dict
        Passed to LogisticRegression (``"balanced"`` handles imbalance).
    random_state : int
        Random seed for reproducibility.
    plot : bool
        If True, display CV metric distributions and threshold tuning plots.

    Returns
    -------
    dict with keys:
        cv_scores      — list of per-fold metric dicts
        mean_metrics   — averaged metrics across folds
        std_metrics    — standard deviations
        best_threshold — optimal threshold (by F1) on OOF predictions
        oof_preds      — out-of-fold probability predictions
        oof_labels     — corresponding true labels
        feature_importance — DataFrame of model coefficients
        model          — full-data fitted LogisticRegression (on scaled data)
        scaler         — fitted StandardScaler (if scale_features=True)
        cv_models      — list of (model, scaler) per fold
    """
    print("=" * 60)
    print("BASELINE MODEL — Logistic Regression")
    print("=" * 60)

    # Separate target if needed
    if y is None:
        if target_col in X.columns:
            y = X[target_col].copy()
            X_feat = X.drop(columns=[target_col])
        else:
            raise ValueError(f"y is None and '{target_col}' not found in X columns.")
    else:
        X_feat = X.copy()

    # Drop any remaining id or metadata columns
    drop_cols = [c for c in ["front_id"] if c in X_feat.columns]
    X_feat = X_feat.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric columns
    X_feat_num = X_feat.select_dtypes(include=[np.number])
    dropped_non_numeric = set(X_feat.columns) - set(X_feat_num.columns)
    if dropped_non_numeric and len(dropped_non_numeric) > 0:
        print(f"  Dropped non-numeric columns: {dropped_non_numeric}")

    n_features = X_feat_num.shape[1]
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    imbalance_ratio = n_negative / n_positive if n_positive > 0 else float("inf")

    print(f"\n  Samples:       {len(y):,}")
    print(f"  Positive:      {n_positive:,}  ({100 * n_positive / len(y):.2f}%)")
    print(f"  Negative:      {n_negative:,}  ({100 * n_negative / len(y):.2f}%)")
    print(f"  Features:      {n_features}")
    print(f"  CV folds:      {n_splits}")
    print(f"  Class weight:  {class_weight}")

    # ── Cross-validation ────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = []
    cv_models = []
    oof_preds = np.empty(len(y))
    oof_preds[:] = np.nan

    print(f"\n  {'Fold':>5s}  {'ROC-AUC':>8s}  {'Gini':>8s}  {'F1':>8s}  {'Prec':>8s}  {'Recall':>8s}  {'Threshold':>9s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat_num, y), 1):
        X_tr, X_val = X_feat_num.iloc[train_idx], X_feat_num.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale
        if scale_features:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
        else:
            scaler = None
            X_tr_s = X_tr.values
            X_val_s = X_val.values

        # Train
        model = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            solver="liblinear",
            n_jobs=1,
        )
        model.fit(X_tr_s, y_tr)
        y_proba = model.predict_proba(X_val_s)[:, 1]

        # Threshold from fold's own data
        fold_thresh_info = evaluate_threshold(y_val, y_proba)
        best_t = fold_thresh_info["threshold"]

        y_pred = (y_proba >= best_t).astype(int)

        roc_auc = roc_auc_score(y_val, y_proba)
        gini = 2 * roc_auc - 1
        f1 = f1_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)

        cv_scores.append({
            "fold": fold,
            "roc_auc": roc_auc,
            "gini": gini,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "threshold": best_t,
            "n_train": len(y_tr),
            "n_val": len(y_val),
        })
        cv_models.append((model, scaler))
        oof_preds[val_idx] = y_proba

        print(f"  {fold:>5d}  {roc_auc:>8.4f}  {gini:>8.4f}  {f1:>8.4f}  {prec:>8.4f}  {rec:>8.4f}  {best_t:>9.4f}")

    # ── Aggregate metrics ───────────────────────────────────────────
    metrics_df = pd.DataFrame(cv_scores)
    mean_m = metrics_df.drop(columns=["fold", "n_train", "n_val"]).mean()
    std_m = metrics_df.drop(columns=["fold", "n_train", "n_val"]).std()

    print(f"\n  ── Mean ────────────────────────────────────────────────")
    print(f"  ROC-AUC:  {mean_m['roc_auc']:.4f}  ± {std_m['roc_auc']:.4f}")
    print(f"  Gini:     {mean_m['gini']:.4f}  ± {std_m['gini']:.4f}")
    print(f"  F1:       {mean_m['f1']:.4f}  ± {std_m['f1']:.4f}")
    print(f"  Precision:{mean_m['precision']:.4f}  ± {std_m['precision']:.4f}")
    print(f"  Recall:   {mean_m['recall']:.4f}  ± {std_m['recall']:.4f}")

    # ── Optimal threshold on OOF ────────────────────────────────────
    oof_thresh = evaluate_threshold(y, oof_preds)
    print(f"\n  Optimal threshold (OOF, by F1): {oof_thresh['threshold']:.4f}"
          f"  (F1={oof_thresh['f1']:.4f}, "
          f"Prec={oof_thresh['precision']:.4f}, "
          f"Recall={oof_thresh['recall']:.4f})")

    # ── Full-data final model ───────────────────────────────────────
    if scale_features:
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X_feat_num)
    else:
        final_scaler = None
        X_scaled = X_feat_num.values

    final_model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver="liblinear",
        n_jobs=1,
    )
    final_model.fit(X_scaled, y)
    final_proba = final_model.predict_proba(X_scaled)[:, 1]
    final_pred = (final_proba >= oof_thresh["threshold"]).astype(int)

    # ── Feature importance (coefficients) ───────────────────────────
    coef_df = pd.DataFrame({
        "feature": X_feat_num.columns,
        "coefficient": final_model.coef_[0],
        "abs_coef": np.abs(final_model.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    print(f"\n  Top 10 coefficients:")
    for _, row in coef_df.head(10).iterrows():
        direction = "+" if row["coefficient"] > 0 else "−"
        print(f"    {direction}  {row['coefficient']:>10.4f}  {row['feature']}")

    # ── Plots ───────────────────────────────────────────────────────
    if plot:
        _plot_baseline_results(metrics_df, oof_thresh, y, oof_preds,
                               final_pred, final_model, X_feat_num, coef_df)

    # ── Final metrics on full train ─────────────────────────────────
    final_roc_auc = roc_auc_score(y, final_proba)
    final_f1 = f1_score(y, final_pred, zero_division=0)
    final_prec = precision_score(y, final_pred, zero_division=0)
    final_rec = recall_score(y, final_pred, zero_division=0)

    print(f"\n  ── Final model (full train) ──")
    print(f"  ROC-AUC:  {final_roc_auc:.4f}")
    print(f"  F1:       {final_f1:.4f}")
    print(f"  Precision:{final_prec:.4f}")
    print(f"  Recall:   {final_rec:.4f}")

    return {
        "cv_scores": cv_scores,
        "mean_metrics": mean_m.to_dict(),
        "std_metrics": std_m.to_dict(),
        "best_threshold": oof_thresh["threshold"],
        "oof_preds": oof_preds,
        "oof_labels": y.values,
        "feature_importance": coef_df,
        "model": final_model,
        "scaler": final_scaler,
        "cv_models": cv_models,
    }


# ===================================================================
# Plotting helper
# ===================================================================

def _plot_baseline_results(metrics_df, oof_thresh, y_true, oof_preds,
                           final_pred, model, X_feat, coef_df):
    """Internal: produce diagnostic plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. CV metric distribution
    ax = axes[0, 0]
    metrics_melt = metrics_df.melt(id_vars=["fold"],
                                   value_vars=["roc_auc", "gini", "f1"],
                                   var_name="metric", value_name="score")
    sns.barplot(data=metrics_melt, x="metric", y="score", ax=ax, palette="Set2",
                edgecolor="gray")
    ax.set_title("Cross-Validation Metrics")
    ax.set_ylabel("Score")

    # 2. ROC curve (OOF)
    from sklearn.metrics import roc_curve
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_true, oof_preds)
    roc_auc = roc_auc_score(y_true, oof_preds)
    ax.plot(fpr, tpr, label=f"OOF ROC-AUC = {roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (OOF Predictions)")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # 3. Precision-Recall & threshold
    ax = axes[0, 2]
    curve = oof_thresh["curve"]
    ax.plot(curve["threshold"], curve["precision"], label="Precision", linewidth=1.5)
    ax.plot(curve["threshold"], curve["recall"], label="Recall", linewidth=1.5)
    ax.plot(curve["threshold"], curve["f1"], label="F1", linewidth=2)
    ax.axvline(oof_thresh["threshold"], color="red", linestyle="--", alpha=0.6,
               label=f"Best F1 = {oof_thresh['threshold']:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning (OOF)")
    ax.legend(loc="best")
    ax.set_xlim(0, 1)

    # 4. Confusion matrix
    ax = axes[1, 0]
    cm = confusion_matrix(y_true, final_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Declined (0)", "Accepted (1)"],
                yticklabels=["Declined (0)", "Accepted (1)"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Full Train)")

    # 5. Top 15 coefficients
    ax = axes[1, 1]
    top15 = coef_df.head(15)
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in top15["coefficient"]]
    ax.barh(range(len(top15)), top15["coefficient"], color=colors, edgecolor="gray")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("Top 15 Feature Coefficients")

    # 6. Probability distribution by class
    ax = axes[1, 2]
    for label_val, color, name in [(0, "#3498db", "Declined"), (1, "#e74c3c", "Accepted")]:
        mask = y_true == label_val
        ax.hist(oof_preds[mask], bins=50, alpha=0.6, color=color,
                label=f"{name} (n={mask.sum()})", density=True)
    ax.axvline(oof_thresh["threshold"], color="purple", linestyle="--",
               label=f"Threshold = {oof_thresh['threshold']:.3f}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("OOF Probability Distribution")
    ax.legend(fontsize=8)

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
    X_tr, X_te, _ = create_features(X_tr, X_te)
    y_tr = train_df["target_value"]

    results = train_baseline(X_tr, y_tr)
    print(f"\nBaseline complete. ROC-AUC: {results['mean_metrics']['roc_auc']:.4f}")
