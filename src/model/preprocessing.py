"""
Preprocessing Module
====================

Handles data cleaning and transformation before modelling:
  - Column selection (drop id, target, high-missing columns)
  - Missing value imputation (median / mode / -999 marker)
  - Outlier clipping (IQR-based capping)
  - Categorical encoding (Target Encoding with smoothing + OOF)
  - Feature scaling (StandardScaler)

All transformations are fit on training data and applied to test data
to prevent data leakage.  The main class is ``Preprocessor``.

Usage
-----
    from src.preprocessing import Preprocessor

    prep = Preprocessor()
    X_train = prep.fit_transform(train, target_col="target_value")
    X_test  = prep.transform(test)
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42

# Columns to drop entirely — not useful as features
COLUMNS_TO_DROP = ["front_id"]

# Categorical columns
CATEGORICAL_COLUMNS = ["db_group_last", "fl_adminarea"]

# Date column — risk of temporal leakage, drop by default
DATE_COLUMN = "decision_day"

# High-missing threshold — columns above this get a binary "is_missing" flag
HIGH_MISSING_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Helper: smoothed target encoding
# ---------------------------------------------------------------------------

def _smooth_target_encoding(series: pd.Series, target: pd.Series,
                            prior: float, alpha: float = 10.0) -> pd.Series:
    """
    Smooth target encoding for a single categorical column.

    encoding = (n_i * mean_i + alpha * prior) / (n_i + alpha)

    Parameters
    ----------
    series       : categorical Series (may contain NaN).
    target       : binary target Series aligned with series.
    prior        : global target mean (float).
    alpha        : smoothing strength (higher → more shrinkage to prior).

    Returns
    -------
    encoded : Series with the same index.
    """
    stats = target.groupby(series).agg(["sum", "count"])
    n_i = stats["count"]
    sum_i = stats["sum"]
    mean_i = sum_i / n_i
    encoded = (n_i * mean_i + alpha * prior) / (n_i + alpha)
    # Fill unseen / NaN categories with prior
    encoded = encoded.reindex(series)
    encoded.index = series.index
    encoded = encoded.fillna(prior)
    return encoded


# ---------------------------------------------------------------------------
# Main Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing pipeline for the overdraft prediction project.

    Fitting requires the target column.  The transform step can be applied
    to both train and test DataFrames.

    Parameters
    ----------
    drop_date_column : bool
        If True, ``decision_day`` is dropped (risk of temporal leakage).
    clip_outliers : bool
        If True, cap extreme values at Q1 - 3*IQR / Q3 + 3*IQR.
    add_missing_flags : bool
        If True, create binary ``<col>_isnan`` flags for high-missing columns.
    scale_features : bool
        If True, apply StandardScaler after imputation / encoding.
    target_encoding_alpha : float
        Smoothing parameter for target encoding (default 10.0).
    n_folds_oof : int
        Number of folds for out-of-fold target encoding on train (0 = no OOF).
        Only applies to ``fit_transform``; ``transform`` always uses full-data maps.
    missing_threshold_high : float
        Columns with missing rate above this get binary flags (default 0.85).
    """

    def __init__(
        self,
        drop_date_column: bool = True,
        clip_outliers: bool = True,
        add_missing_flags: bool = True,
        scale_features: bool = False,  # False by default (tree models don't need it)
        target_encoding_alpha: float = 10.0,
        n_folds_oof: int = 5,
        missing_threshold_high: float = HIGH_MISSING_THRESHOLD,
    ):
        self.drop_date_column = drop_date_column
        self.clip_outliers = clip_outliers
        self.add_missing_flags = add_missing_flags
        self.scale_features = scale_features
        self.target_encoding_alpha = target_encoding_alpha
        self.n_folds_oof = n_folds_oof
        self.missing_threshold_high = missing_threshold_high

        # Learned state — set during fit
        self.numeric_cols_ = []
        self.cat_cols_ = []
        self.high_missing_cols_ = []
        self.median_vals_ = {}
        self.mode_vals_ = {}
        self.clip_bounds_ = {}
        self.target_encoding_maps_ = {}
        self.target_prior_ = None
        self.scaler_ = None
        self.oof_encodings_ = {}  # fitted on full data for transform()
        self._fitted = False

    # -------- Fit -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Learn imputation values, encoding maps, scaling parameters from train.

        Parameters
        ----------
        X : pd.DataFrame
            Training features (including target if requested).
        y : pd.Series or None
            Target series.  Required if categorical columns need target encoding
            or if target column is inside X.

        Returns
        -------
        self
        """
        df = X.copy()

        # Separate target if present in X
        if y is None and "target_value" in df.columns:
            y = df["target_value"]
            df = df.drop(columns=["target_value"])

        # Drop id columns
        for col in COLUMNS_TO_DROP:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Drop date column
        if self.drop_date_column and DATE_COLUMN in df.columns:
            df.drop(columns=[DATE_COLUMN], inplace=True)

        # Identify numeric vs categorical
        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

        # Add any other object columns as categorical
        for c in df.select_dtypes(include="object").columns:
            if c not in self.cat_cols_:
                self.cat_cols_.append(c)

        # Remove cat cols that happen to be numeric-like
        self.cat_cols_ = [c for c in self.cat_cols_ if c in df.columns]

        # ---- High-missing columns (> threshold) ----
        if self.add_missing_flags:
            miss_rate = df[self.numeric_cols_].isnull().mean()
            self.high_missing_cols_ = miss_rate[miss_rate > self.missing_threshold_high].index.tolist()
            if self.high_missing_cols_:
                print(f"  High-missing columns ({self.missing_threshold_high:.0%}): "
                      f"{self.high_missing_cols_}")
        else:
            self.high_missing_cols_ = []

        # ---- Imputation values ----
        for col in self.numeric_cols_:
            self.median_vals_[col] = df[col].median()

        for col in self.cat_cols_:
            mode_series = df[col].mode(dropna=True)
            self.mode_vals_[col] = mode_series.iloc[0] if not mode_series.empty else "__MISSING__"

        # ---- Outlier bounds ----
        if self.clip_outliers:
            for col in self.numeric_cols_:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                self.clip_bounds_[col] = (lower, upper)

        # ---- Target encoding maps (full-data, for transform) ----
        if y is not None and len(self.cat_cols_) > 0:
            self.target_prior_ = y.mean()
            for col in self.cat_cols_:
                encoding = _smooth_target_encoding(
                    df[col], y, prior=self.target_prior_, alpha=self.target_encoding_alpha
                )
                self.target_encoding_maps_[col] = encoding

        # ---- Scaler ----
        if self.scale_features:
            # Fit on imputed + encoded numeric data
            processed = self._transform_core(df, y_for_oof=None, is_train=True)
            num_for_scaler = processed[self.numeric_cols_].select_dtypes(include=[np.number])
            self.scaler_ = StandardScaler().fit(num_for_scaler)

        self._fitted = True
        return self

    # -------- Transform -----------------------------------------------------

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Apply learned transformations.  Can be used on train or test.

        For training data, call ``fit_transform`` instead to get OOF encodings.
        For test / unseen data, this method uses full-data encoding maps.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        df = self._transform_core(X.copy(), y_for_oof=None, is_train=False)

        if self.scale_features and self.scaler_ is not None:
            num_cols = [c for c in self.numeric_cols_ if c in df.columns]
            df[num_cols] = self.scaler_.transform(df[num_cols])

        return df

    # -------- Fit-transform (with OOF encoding) -----------------------------

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform training data using OOF target encoding
        to prevent data leakage for categorical features.
        """
        df = X.copy()

        # Separate target — always remove from feature matrix
        if y is None and "target_value" in df.columns:
            y = df["target_value"]
            df = df.drop(columns=["target_value"])
        elif y is not None and "target_value" in df.columns:
            df = df.drop(columns=["target_value"])

        # Drop id / date
        for col in COLUMNS_TO_DROP + ([DATE_COLUMN] if self.drop_date_column else []):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Identify columns
        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        for c in df.select_dtypes(include="object").columns:
            if c not in self.cat_cols_:
                self.cat_cols_.append(c)
        self.cat_cols_ = [c for c in self.cat_cols_ if c in df.columns]

        # High-missing flags
        if self.add_missing_flags:
            miss_rate = df[self.numeric_cols_].isnull().mean()
            self.high_missing_cols_ = miss_rate[miss_rate > self.missing_threshold_high].index.tolist()
        else:
            self.high_missing_cols_ = []

        # Imputation values, outlier bounds, scaler
        self._fit_imputation_and_bounds(df)

        # Target prior
        if y is not None:
            self.target_prior_ = y.mean()
            # Full-data encoding map for transform()
            for col in self.cat_cols_:
                self.target_encoding_maps_[col] = _smooth_target_encoding(
                    df[col], y, prior=self.target_prior_, alpha=self.target_encoding_alpha
                )

        # ---- Out-of-fold target encoding for train ----
        if y is not None and len(self.cat_cols_) > 0 and self.n_folds_oof > 1:
            print(f"  Applying {self.n_folds_oof}-fold OOF target encoding for "
                  f"{len(self.cat_cols_)} categorical column(s): {self.cat_cols_}")
            oof_encoded = pd.Series(index=df.index, dtype=float)
            skf = StratifiedKFold(n_splits=self.n_folds_oof, shuffle=True, random_state=RANDOM_STATE)

            for col in self.cat_cols_:
                oof_vals = np.empty(len(df))
                oof_vals[:] = np.nan
                for train_idx, val_idx in skf.split(df, y):
                    y_fold = y.iloc[train_idx]
                    X_fold = df[col].iloc[train_idx]
                    X_val = df[col].iloc[val_idx]
                    encoding_fold = _smooth_target_encoding(
                        X_fold, y_fold, prior=self.target_prior_, alpha=self.target_encoding_alpha
                    )
                    # Map to validation indices
                    fold_map = encoding_fold.groupby(X_fold).first()  # one value per category
                    oof_vals[val_idx] = X_val.map(fold_map).fillna(self.target_prior_).values
                df[f"{col}_te"] = oof_vals

            # Update numeric cols to include the new TE columns
            for col in self.cat_cols_:
                self.numeric_cols_.append(f"{col}_te")
        elif y is not None and len(self.cat_cols_) > 0:
            # No OOF — apply smoothed encoding directly
            for col in self.cat_cols_:
                df[f"{col}_te"] = self.target_encoding_maps_[col]

        # Drop raw categorical columns
        df.drop(columns=[c for c in self.cat_cols_ if c in df.columns], inplace=True, errors="ignore")

        # ---- Core transforms (impute, clip) ----
        df = self._apply_core_transforms(df)

        # ---- Scale ----
        if self.scale_features:
            num_cols = [c for c in self.numeric_cols_ if c in df.columns]
            self.scaler_ = StandardScaler().fit(df[num_cols])
            df[num_cols] = self.scaler_.transform(df[num_cols])

        self._fitted = True
        return df

    # -------- Internal helpers ----------------------------------------------

    def _fit_imputation_and_bounds(self, df: pd.DataFrame):
        """Learn median, mode, and clip bounds from training data."""
        for col in self.numeric_cols_:
            self.median_vals_[col] = df[col].median()
        for col in self.cat_cols_:
            mode_series = df[col].mode(dropna=True)
            self.mode_vals_[col] = mode_series.iloc[0] if not mode_series.empty else "__MISSING__"
        if self.clip_outliers:
            for col in self.numeric_cols_:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                self.clip_bounds_[col] = (q1 - 3.0 * iqr, q3 + 3.0 * iqr)

    def _transform_core(self, df: pd.DataFrame, y_for_oof=None, is_train: bool = False) -> pd.DataFrame:
        """Apply core transformations without OOF encoding."""
        # Drop id / date
        for col in COLUMNS_TO_DROP + ([DATE_COLUMN] if self.drop_date_column else []):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Target encode categoricals using full-data maps
        if self.target_encoding_maps_:
            for col in self.cat_cols_:
                if col in df.columns:
                    df[f"{col}_te"] = df[col].map(
                        self.target_encoding_maps_[col]
                    ).fillna(self.target_prior_ or 0.0)
            df.drop(columns=[c for c in self.cat_cols_ if c in df.columns], inplace=True, errors="ignore")
            # Add TE columns to numeric cols if not already
            for col in self.cat_cols_:
                te_col = f"{col}_te"
                if te_col in df.columns and te_col not in self.numeric_cols_:
                    self.numeric_cols_.append(te_col)

        df = self._apply_core_transforms(df)
        return df

    def _apply_core_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values and clip outliers."""
        # Binary flags for high-missing columns
        if self.add_missing_flags:
            for col in self.high_missing_cols_:
                if col in df.columns:
                    df[f"{col}_isnan"] = df[col].isnull().astype(int)

        # Impute numeric — median
        for col in self.numeric_cols_:
            if col in df.columns:
                med = self.median_vals_.get(col)
                if med is not None:
                    df[col] = df[col].fillna(med)

        # Impute categorical — mode
        for col in self.cat_cols_:
            if col in df.columns:
                mode_val = self.mode_vals_.get(col)
                if mode_val is not None:
                    df[col] = df[col].fillna(mode_val)

        # Clip outliers
        if self.clip_outliers:
            for col in self.numeric_cols_:
                if col in df.columns and col in self.clip_bounds_:
                    lo, hi = self.clip_bounds_[col]
                    df[col] = df[col].clip(lower=lo, upper=hi)

        return df

    # -------- Convenience ---------------------------------------------------

    def get_feature_names(self) -> list:
        """Return list of feature names after preprocessing (excludes target)."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted first.")
        exclude = {"target_value", "target"}
        feats = set()
        for col in self.numeric_cols_:
            if col not in exclude:
                feats.add(col)
        for col in self.high_missing_cols_:
            feats.add(f"{col}_isnan")
        for col in self.cat_cols_:
            te_col = f"{col}_te"
            feats.add(te_col)
        return sorted(f for f in feats if f not in exclude)


# ===================================================================
# Convenience function (notebook-friendly)
# ===================================================================

def preprocess_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "target_value",
    drop_date_column: bool = True,
    clip_outliers: bool = True,
    add_missing_flags: bool = True,
    scale_features: bool = False,
    target_encoding_alpha: float = 10.0,
    n_folds_oof: int = 5,
) -> tuple:
    """
    One-call convenience: fit preprocessor on train, transform train & test.

    Returns
    -------
    (X_train, X_test, preprocessor)
    """
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    y_train = train[target_col].copy()

    prep = Preprocessor(
        drop_date_column=drop_date_column,
        clip_outliers=clip_outliers,
        add_missing_flags=add_missing_flags,
        scale_features=scale_features,
        target_encoding_alpha=target_encoding_alpha,
        n_folds_oof=n_folds_oof,
    )

    print("\nFitting preprocessor on train...")
    X_train = prep.fit_transform(train, y=y_train)
    print(f"  X_train shape: {X_train.shape}")

    print("\nTransforming test...")
    X_test = prep.transform(test)
    print(f"  X_test shape:  {X_test.shape}")

    print(f"\nFeatures ({len(prep.get_feature_names())} total):")
    print(f"  {prep.get_feature_names()}")

    return X_train, X_test, prep


# ===================================================================
# Standalone entry point
# ===================================================================
if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train_apps.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_apps.csv"))
    X_tr, X_te, _ = preprocess_data(train_df, test_df)
    print(f"\nPreprocessing complete. X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
