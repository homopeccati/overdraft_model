"""
Feature Selection Module
========================

Analyses feature importance using multiple methods:
  1. Gain-based importance (from the trained model)
  2. Permutation importance (model-agnostic)
  3. SHAP values (game-theoretic attribution)
  4. Multicollinearity check (VIF)

Returns a list of selected (retained) features.

Usage
-----
    from src.model.feature_selection import analyze_features

    selected = analyze_features(model_results, X_train, y_train)
    # selected["features"] -> list of column names to keep
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import shap
    _shap_available = True
except ImportError:
    shap = None
    _shap_available = False

RANDOM_STATE = 42
VIF_THRESHOLD = 10          # features with VIF > 10 considered highly collinear
SHAP_SAMPLE_SIZE = 500      # rows used for SHAP (speed / memory)
PERMUTATION_REPEATS = 5     # repeats for permutation importance

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# ===================================================================
# 1. Gain-based importance
# ===================================================================

def _gain_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract built-in feature importance from a fitted model."""
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        # Normalise to [0, 1]
        fi_sum = fi.sum()
        if fi_sum > 0:
            fi = fi / fi_sum
        return pd.DataFrame({
            "feature": feature_names,
            "gain_importance": fi,
        }).sort_values("gain_importance", ascending=False).reset_index(drop=True)
    return None


# ===================================================================
# 2. Permutation importance
# ===================================================================

def _permutation_importance(model, X: pd.DataFrame, y: pd.Series,
                            repeats: int = PERMUTATION_REPEATS,
                            sample_size: int = 5000) -> pd.DataFrame:
    """
    Compute permutation importance on a sample.
    Uses ROC-AUC drop as the scoring metric.
    """
    if len(X) > sample_size:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y

    result = permutation_importance(
        model, X_sample, y_sample,
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": X.columns,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    return df


# ===================================================================
# 3. SHAP analysis
# ===================================================================

def _shap_analysis(model, X: pd.DataFrame, sample_size: int = SHAP_SAMPLE_SIZE) -> dict:
    """
    Compute SHAP values on a sample of the data.
    Returns a dict with: shap_values, feature_importance, X_sample.
    """
    if len(X) > sample_size:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx].copy()
    else:
        X_sample = X.copy()

    if not _shap_available:
        print("  SHAP not installed — skipping.  Install with: pip install shap")
        return None

    try:
        # Use TreeExplainer for tree models, generic for others
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)
        except Exception:
            # Fallback to generic explainer
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

        # Mean absolute SHAP as global importance
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X_sample.columns,
            "shap_importance": mean_abs_shap,
        }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

        return {
            "shap_values": shap_values,
            "importance_df": importance_df,
            "X_sample": X_sample,
            "explainer": explainer,
        }
    except Exception as e:
        print(f"  SHAP analysis skipped: {e}")
        return None


def _plot_shap_summary(shap_result: dict, max_features: int = 20):
    """SHAP summary (beeswarm) and bar plots."""
    if shap_result is None:
        return

    shap_values = shap_result["shap_values"]
    X_sample = shap_result["X_sample"]
    imp_df = shap_result["importance_df"]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * max_features)))

    # Bar plot
    ax = axes[0]
    top = imp_df.head(max_features)
    colors = sns.color_palette("viridis", len(top))
    ax.barh(range(len(top)), top["shap_importance"], color=colors, edgecolor="gray")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("SHAP Feature Importance")

    # Beeswarm
    ax = axes[1]
    shap.summary_plot(
        shap_values.values[:, :max_features],
        X_sample.iloc[:, :max_features],
        plot_type="dot",
        show=False,
        max_display=max_features,
        alpha=0.6,
    )
    ax.set_title("SHAP Summary (beeswarm)")

    plt.tight_layout()
    plt.show()


def _plot_shap_waterfall(shap_result: dict, index: int = 0):
    """Waterfall plot for a single observation."""
    if shap_result is None:
        return
    shap_values = shap_result["shap_values"]
    if index < len(shap_values):
        plt.figure(figsize=(8, 4))
        shap.waterfall_plot(shap_values[index], max_display=15, show=False)
        plt.title(f"SHAP Waterfall — Sample #{index}")
        plt.tight_layout()
        plt.show()


# ===================================================================
# 4. Multicollinearity (VIF)
# ===================================================================

def _compute_vif(df: pd.DataFrame, threshold: float = VIF_THRESHOLD) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.
    Uses the matrix-inversion method (works on standardised data).
    Features with VIF > threshold are considered highly collinear.
    """
    from sklearn.linear_model import LinearRegression

    vif_data = []
    cols = df.columns

    for col in cols:
        other_cols = [c for c in cols if c != col]
        if len(other_cols) == 0:
            vif_data.append({"feature": col, "vif": 1.0})
            continue

        # Regress col against all other features
        lr = LinearRegression()
        lr.fit(df[other_cols], df[col])
        r2 = lr.score(df[other_cols], df[col])

        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        vif_data.append({"feature": col, "vif": vif})

    vif_df = pd.DataFrame(vif_data).sort_values("vif", ascending=False)
    return vif_df


# ===================================================================
# 5. Combined analysis
# ===================================================================

def analyze_features(
    model_results: dict,
    X: pd.DataFrame,
    y: pd.Series = None,
    target_col: str = "target_value",
    model_name: str = None,
    vif_threshold: float = VIF_THRESHOLD,
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run full feature importance analysis and return selected features.

    Parameters
    ----------
    model_results : dict
        Output from ``train_models()`` — must contain ``results`` and ``best_model`` keys.
    X : pd.DataFrame
        Feature matrix **or** full training DataFrame (if ``y`` is None).
    y : pd.Series or None
        Target labels.
    target_col : str
        Column name for the target (used when ``y`` is None).
    model_name : str or None
        Which model to analyse.  If None, uses the best model from ``model_results``.
    vif_threshold : float
        Maximum VIF for a feature to be retained.
    plot : bool
        If True, display SHAP plots.
    verbose : bool
        If True, print detailed results.

    Returns
    -------
    dict with keys:
        selected_features — list of feature names to keep
        dropped_features  — list of dropped feature names (low importance / high VIF)
        importance_df     — combined importance rankings
        vif_df            — VIF results
        shap_result       — SHAP analysis output
        model_name        — name of the analysed model
    """
    print("=" * 60)
    print("FEATURE SELECTION")
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
    dropped_non_num = set(X_feat.columns) - set(X_num.columns)
    if dropped_non_num and verbose:
        print(f"  Dropped non-numeric: {dropped_non_num}")

    # Determine which model to analyse
    if model_name is None:
        model_name = model_results.get("best_model", list(model_results["results"].keys())[0])

    model_entry = model_results["results"].get(model_name)
    if model_entry is None:
        available = list(model_results["results"].keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    model = model_entry["model"]
    feature_names = list(X_num.columns)

    if verbose:
        print(f"\n  Analysing model: {model_name}")
        print(f"  Features:        {len(feature_names)}")
        print(f"  Samples:         {len(y):,}")

    # ── 1. Gain importance ──────────────────────────────────────────
    if verbose:
        print("\n  ── Gain-based importance ──")
    gain_df = _gain_importance(model, feature_names)

    if gain_df is not None and verbose:
        for _, row in gain_df.head(10).iterrows():
            print(f"    {row['gain_importance']:.4f}  {row['feature']}")

    # ── 2. Permutation importance ───────────────────────────────────
    if verbose:
        print("\n  ── Permutation importance ──")
    perm_df = _permutation_importance(model, X_num, y)

    if verbose:
        for _, row in perm_df.head(10).iterrows():
            print(f"    {row['perm_importance_mean']:.6f}  ± {row['perm_importance_std']:.6f}  {row['feature']}")

    # ── 3. SHAP analysis ────────────────────────────────────────────
    if verbose:
        print(f"\n  ── SHAP analysis (sample={SHAP_SAMPLE_SIZE}) ──")

    shap_result = _shap_analysis(model, X_num)

    if shap_result is not None and verbose:
        for _, row in shap_result["importance_df"].head(10).iterrows():
            print(f"    {row['shap_importance']:.6f}  {row['feature']}")

    if plot and shap_result is not None:
        _plot_shap_summary(shap_result)
        _plot_shap_waterfall(shap_result, index=0)

    # ── 4. Combine importance rankings ──────────────────────────────
    combined = gain_df.copy() if gain_df is not None else perm_df.copy()
    if gain_df is not None:
        combined = combined.merge(perm_df, on="feature", how="outer")
    if shap_result is not None:
        combined = combined.merge(shap_result["importance_df"], on="feature", how="outer")

    combined = combined.fillna(0)

    # Rank features by average of available importance metrics
    rank_cols = []
    if "gain_importance" in combined.columns:
        rank_cols.append("gain_importance")
    if "perm_importance_mean" in combined.columns:
        rank_cols.append("perm_importance_mean")
    if "shap_importance" in combined.columns:
        rank_cols.append("shap_importance")

    if rank_cols:
        combined["avg_importance"] = combined[rank_cols].mean(axis=1)
        combined = combined.sort_values("avg_importance", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n  ── Combined ranking (top 15) ──")
        display_cols = ["feature"] + rank_cols + (["avg_importance"] if "avg_importance" in combined.columns else [])
        print(combined.head(15)[display_cols].to_string(float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    # ── 5. VIF analysis ─────────────────────────────────────────────
    if verbose:
        print(f"\n  ── Multicollinearity (VIF > {vif_threshold}) ──")
    vif_df = _compute_vif(X_num)

    high_vif = vif_df[vif_df["vif"] > vif_threshold]
    if verbose:
        print(f"  Features with high VIF: {len(high_vif)}")
        for _, row in high_vif.iterrows():
            print(f"    VIF={row['vif']:.2f}  {row['feature']}")

    # ── 6. Select features ──────────────────────────────────────────
    # Drop features with:
    #   a) zero importance across all metrics
    #   b) VIF > threshold (keep the one with higher importance among correlated pairs)
    zero_imp = set()
    if "avg_importance" in combined.columns:
        zero_imp = set(combined[combined["avg_importance"] <= 0]["feature"])

    # For high-VIF features, keep the one with highest importance per correlated group
    # Simple approach: drop all high-VIF features except top-N by importance
    high_vif_features = set(high_vif["feature"])
    # Remove high-VIF features that are in zero-importance already
    to_drop_high_vif = high_vif_features - zero_imp

    # Among high-VIF, keep those with highest importance
    if to_drop_high_vif and "avg_importance" in combined.columns:
        high_vif_ranked = combined[combined["feature"].isin(to_drop_high_vif)].sort_values("avg_importance", ascending=False)
        # Keep top 30% of high-VIF features (those most important)
        keep_count = max(1, int(len(high_vif_ranked) * 0.3))
        features_to_keep = set(high_vif_ranked.head(keep_count)["feature"])
        to_drop_high_vif = to_drop_high_vif - features_to_keep

    dropped = zero_imp | to_drop_high_vif
    selected = [f for f in feature_names if f not in dropped]

    if verbose:
        print(f"\n  ── Selection summary ──")
        print(f"  Total features:      {len(feature_names)}")
        print(f"  Zero importance:     {len(zero_imp)}")
        print(f"  High VIF dropped:    {len(to_drop_high_vif)}")
        print(f"  Selected features:   {len(selected)}")
        if dropped:
            print(f"  Dropped ({len(dropped)}): {sorted(dropped)}")

    return {
        "selected_features": selected,
        "dropped_features": sorted(dropped),
        "importance_df": combined,
        "vif_df": vif_df,
        "shap_result": shap_result,
        "model_name": model_name,
    }


# ===================================================================
# Standalone entry point
# ===================================================================
if __name__ == "__main__":
    import os
    from src.model.preprocessing import preprocess_data
    from src.model.features import create_features
    from src.model.models import train_models

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train_apps.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_apps.csv"))

    X_tr, X_te, _ = preprocess_data(train_df, test_df)
    X_tr, X_te, _ = create_features(X_tr, X_te, verbose=False)
    y_tr = train_df["target_value"]

    model_results = train_models(X_tr, y_tr, plot=False)
    analysis = analyze_features(model_results, X_tr, y_tr, plot=False)
    print(f"\nSelected {len(analysis['selected_features'])} / {len(X_tr.columns)} features")
