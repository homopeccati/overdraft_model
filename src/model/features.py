"""
Feature Engineering Module
==========================

Creates derived features from preprocessed data (no missing values,
outliers already capped).

Priority features (from project plan):
  1. rate_spread           offered_rate - cb_rate
  2. limit_request_ratio   loan_amount_last / overdraft_limit_max
  3. limit_range           overdraft_limit_max - overdraft_limit_min
  4. activity_ratio_30_90  cnt_deb_ul_ip_30 / (cnt_deb_ul_ip_90 + 1)
  5. log_balance           log1p-transformed balance (handles negatives)

Additional features:
  6. activity_decay_90_30  cnt_deb_ul_ip_90 - cnt_deb_ul_ip_30
  7. sum_deb_total         sum_deb_ul_90 + sum_deb_ul_30
  8. payment_to_income     cnt_deb_loan_90 / (cnt_cred_loan_90 + 1)
  9. loan_amount_ratio     loan_amount_last / (overdraft_limit_min + eps)

Usage
-----
    from src.model.features import create_features

    X_train_aug, X_test_aug, feat_meta = create_features(X_train, X_test)
"""

import pandas as pd
import numpy as np

EPS = 1e-6

# Human-readable descriptions for each engineered feature
FEATURE_DESCRIPTIONS = {
    "rate_spread":             "Offer rate spread over central bank key rate",
    "limit_request_ratio":     "Requested amount / max offered limit",
    "limit_range":             "Width of offered limit corridor (max - min)",
    "activity_ratio_30_90":    "Recent vs broader payment activity ratio",
    "log_balance":             "Log-transformed min. account balance",
    "activity_decay_90_30":    "Payment activity decay (90d - 30d)",
    "sum_deb_total":           "Total transfer volume (90d + 30d)",
    "payment_to_income":       "Loan repayments per received loan",
    "loan_amount_ratio":       "Requested amount / min offered limit",
}

# Set of all required source columns for feature creation
REQUIRED_COLUMNS = {
    "offered_rate",
    "cb_rate",
    "loan_amount_last",
    "overdraft_limit_max",
    "overdraft_limit_min",
    "cnt_deb_ul_ip_30",
    "cnt_deb_ul_ip_90",
    "balance_rur_amt_30_min",
    "sum_deb_ul_90",
    "sum_deb_ul_30",
    "cnt_deb_loan_90",
    "cnt_cred_loan_90",
}


def _safe_ratio(num: pd.Series, den: pd.Series, eps: float = EPS) -> pd.Series:
    """Compute num / den, clipping denom away from zero to avoid division issues."""
    safe_den = np.where(np.abs(den) < eps, eps, den)
    return num / safe_den


def create_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, verbose: bool = True
) -> tuple:
    """
    Create engineered features on preprocessed data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Preprocessed training features (no missing values).
    X_test : pd.DataFrame
        Preprocessed test features (no missing values).
    verbose : bool
        If True, print summary of created features.

    Returns
    -------
    (X_train_aug, X_test_aug, feat_meta)
        Augmented DataFrames and a dict with:
          - 'feature_names': list of created feature column names
          - 'descriptions': dict mapping feature name to description
          - 'source_cols': list of source columns used
    """
    if verbose:
        print("=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Warn about missing source columns
    missing_src = REQUIRED_COLUMNS - set(X_tr.columns)
    if missing_src and verbose:
        print(f"  Note: some source columns not found: {missing_src}")

    created_features = []

    # ── 1. rate_spread ────────────────────────────────────────────────
    if {"offered_rate", "cb_rate"}.issubset(X_tr.columns):
        X_tr["rate_spread"] = X_tr["offered_rate"] - X_tr["cb_rate"]
        X_te["rate_spread"] = X_te["offered_rate"] - X_te["cb_rate"]
        created_features.append("rate_spread")

    # ── 2. limit_request_ratio ────────────────────────────────────────
    if {"loan_amount_last", "overdraft_limit_max"}.issubset(X_tr.columns):
        X_tr["limit_request_ratio"] = _safe_ratio(
            X_tr["loan_amount_last"], X_tr["overdraft_limit_max"]
        )
        X_te["limit_request_ratio"] = _safe_ratio(
            X_te["loan_amount_last"], X_te["overdraft_limit_max"]
        )
        created_features.append("limit_request_ratio")

    # ── 3. limit_range ────────────────────────────────────────────────
    if {"overdraft_limit_max", "overdraft_limit_min"}.issubset(X_tr.columns):
        X_tr["limit_range"] = X_tr["overdraft_limit_max"] - X_tr["overdraft_limit_min"]
        X_te["limit_range"] = X_te["overdraft_limit_max"] - X_te["overdraft_limit_min"]
        created_features.append("limit_range")

    # ── 4. activity_ratio_30_90 ───────────────────────────────────────
    if {"cnt_deb_ul_ip_30", "cnt_deb_ul_ip_90"}.issubset(X_tr.columns):
        X_tr["activity_ratio_30_90"] = (
            X_tr["cnt_deb_ul_ip_30"] / (X_tr["cnt_deb_ul_ip_90"] + 1.0)
        )
        X_te["activity_ratio_30_90"] = (
            X_te["cnt_deb_ul_ip_30"] / (X_te["cnt_deb_ul_ip_90"] + 1.0)
        )
        created_features.append("activity_ratio_30_90")

    # ── 5. log_balance ────────────────────────────────────────────────
    if "balance_rur_amt_30_min" in X_tr.columns:
        # Handle negative values by shifting: log1p(x - min) so min becomes 0
        # This avoids NaN from log of negative numbers
        bal = X_tr["balance_rur_amt_30_min"]
        bal_min = bal.min()
        if bal_min < 0:
            X_tr["log_balance"] = np.log1p(bal - bal_min)
            # Apply same shift to test (using train's min)
            X_te["log_balance"] = np.log1p(X_te["balance_rur_amt_30_min"] - bal_min)
            if verbose:
                print(f"  log_balance: shifted by {-bal_min:.4f} (train min) before log1p")
        else:
            X_tr["log_balance"] = np.log1p(bal)
            X_te["log_balance"] = np.log1p(X_te["balance_rur_amt_30_min"])
        created_features.append("log_balance")

    # ── 6. activity_decay_90_30 ───────────────────────────────────────
    if {"cnt_deb_ul_ip_90", "cnt_deb_ul_ip_30"}.issubset(X_tr.columns):
        X_tr["activity_decay_90_30"] = (
            X_tr["cnt_deb_ul_ip_90"] - X_tr["cnt_deb_ul_ip_30"]
        )
        X_te["activity_decay_90_30"] = (
            X_te["cnt_deb_ul_ip_90"] - X_te["cnt_deb_ul_ip_30"]
        )
        created_features.append("activity_decay_90_30")

    # ── 7. sum_deb_total ──────────────────────────────────────────────
    if {"sum_deb_ul_90", "sum_deb_ul_30"}.issubset(X_tr.columns):
        X_tr["sum_deb_total"] = X_tr["sum_deb_ul_90"] + X_tr["sum_deb_ul_30"]
        X_te["sum_deb_total"] = X_te["sum_deb_ul_90"] + X_te["sum_deb_ul_30"]
        created_features.append("sum_deb_total")

    # ── 8. payment_to_income ──────────────────────────────────────────
    if {"cnt_deb_loan_90", "cnt_cred_loan_90"}.issubset(X_tr.columns):
        X_tr["payment_to_income"] = X_tr["cnt_deb_loan_90"] / (X_tr["cnt_cred_loan_90"] + 1.0)
        X_te["payment_to_income"] = X_te["cnt_deb_loan_90"] / (X_te["cnt_cred_loan_90"] + 1.0)
        created_features.append("payment_to_income")

    # ── 9. loan_amount_ratio ──────────────────────────────────────────
    if {"loan_amount_last", "overdraft_limit_min"}.issubset(X_tr.columns):
        X_tr["loan_amount_ratio"] = _safe_ratio(
            X_tr["loan_amount_last"], X_tr["overdraft_limit_min"]
        )
        X_te["loan_amount_ratio"] = _safe_ratio(
            X_te["loan_amount_last"], X_te["overdraft_limit_min"]
        )
        created_features.append("loan_amount_ratio")

    # ── Summary ───────────────────────────────────────────────────────
    feat_meta = {
        "feature_names": created_features,
        "descriptions": {f: FEATURE_DESCRIPTIONS.get(f, "") for f in created_features},
        "source_cols": list(REQUIRED_COLUMNS & set(X_train.columns)),
    }

    if verbose:
        print(f"\n  Features created ({len(created_features)}):")
        for name in created_features:
            desc = FEATURE_DESCRIPTIONS.get(name, "")
            print(f"    {name:30s}  {desc}")
        print(f"\n  X_train shape: {X_tr.shape}")
        print(f"  X_test shape:  {X_te.shape}")

    return X_tr, X_te, feat_meta


# ===================================================================
# Standalone entry point
# ===================================================================
if __name__ == "__main__":
    import os
    from src.model.preprocessing import preprocess_data

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train_apps.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_apps.csv"))

    X_tr, X_te, prep = preprocess_data(train_df, test_df)
    X_tr_aug, X_te_aug, meta = create_features(X_tr, X_te)
    print(f"\nFeature engineering complete. {len(meta['feature_names'])} features added.")
