"""
Exploratory Data Analysis (EDA) Module
=======================================

Provides functions for:
  - Data structure overview (size, types, missing values)
  - Target variable analysis (class balance)
  - Numerical feature distributions and outlier detection
  - Categorical variable analysis
  - Correlation analysis with the target (point-biserial, Cramér's V)
  - Summary report generation

All functions are designed to be importable and callable from
notebooks/pipeline.ipynb.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Human-readable labels for display
# ---------------------------------------------------------------------------

COLUMN_LABELS = {
    "front_id":                              "Application ID",
    "decision_day":                          "Decision date",
    "loan_amount_last":                      "Requested loan amount",
    "overdraft_limit_min":                   "Min. overdraft limit (pre-scoring)",
    "overdraft_limit_max":                   "Max. overdraft limit (pre-scoring)",
    "offered_rate":                          "Offered interest rate",
    "cb_rate":                               "Central bank key rate",
    "corp_credit_products":                  "Corp. credit product events",
    "sum_deb_ul_90":                         "Transfers to legal entities, 90d",
    "sum_deb_ul_30":                         "Transfers to legal entities, 30d",
    "cnt_deb_loan_90":                       "Loan repayments, 90d",
    "cnt_deb_ul_ip_90":                      "Transfers to legal entities & IP, 90d",
    "cnt_deb_ul_ip_30":                      "Transfers to legal entities & IP, 30d",
    "balance_rur_amt_30_min":                "Min. RUB account balance, 30d",
    "cnt_cred_loan_90":                      "Loans received, 90d",
    "loan_rev_max_start_non_fin":            "Months to max start date (active revolving)",
    "loan_rev_min_start_fin":                "Months to min start date (closed revolving)",
    "app_term_mean_360":                     "Avg. loan application term, 360d",
    "overdraft_app_term_max_360":            "Max. overdraft app term, 360d",
    "days_from_authperson_registration":     "Days since manager registration",
    "fl_hdb_bki_total_active_products":      "Active BKI credit products",
    "corp_list":                             "Corp. list events",
    "count_all_corp_dashboard_events":       "Corp. online banking actions",
    "p75_time_spent_minutes":                "75th pctl time in banking app",
    "sum_deb_investment_90":                 "Investments & deposits, 90d",
    "db_group_last":                         "Last credit product type",
    "fl_adminarea":                          "Client region",
    "target_value":                          "Target: accepted (1) / declined (0)",
}


def label(col: str) -> str:
    """Return human-readable label for a column, or the column name as fallback."""
    return COLUMN_LABELS.get(col, col)


def _fmt_labels(names: list) -> str:
    """Format a list of column names with their labels for display."""
    parts = []
    for n in names:
        lbl = label(n)
        if lbl == n:
            parts.append(n)
        else:
            parts.append(f"{n}  ({lbl})")
    return ", ".join(parts)


# ===================================================================
# 1. Data structure overview
# ===================================================================

def overview(train: pd.DataFrame, test: pd.DataFrame, dataset_name: str = "train_apps.csv") -> dict:
    """
    Print and return a structural summary of the datasets.

    Returns
    -------
    dict with keys:
        'train_shape', 'test_shape', 'train_dtypes', 'train_missing',
        'test_missing', 'train_duplicates', 'test_duplicates',
        'numeric_cols', 'categorical_cols', 'id_col'
    """
    print("=" * 60)
    print("DATA STRUCTURE OVERVIEW")
    print("=" * 60)

    # Shapes
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape:  {test.shape}")

    # Dtypes
    print("\n--- Train dtypes ---")
    dtype_counts = train.dtypes.value_counts()
    for dt, cnt in dtype_counts.items():
        print(f"  {dt}: {cnt} columns")

    # Missing values
    train_miss = train.isnull().sum()
    train_miss_pct = 100 * train_miss / len(train)
    test_miss = test.isnull().sum()
    test_miss_pct = 100 * test_miss / len(test)

    miss_df = pd.DataFrame({
        "Train Missings": train_miss,
        "Train %": train_miss_pct.round(2),
        "Test Missings": test_miss,
        "Test %": test_miss_pct.round(2),
    })
    miss_df = miss_df[miss_df["Train Missings"] > 0].sort_values("Train Missings", ascending=False)
    print(f"\n--- Missing values (columns with any missing in train) ---")
    print(f"Total columns with missings in train: {len(miss_df)}")
    # Print with descriptive labels
    for col in miss_df.index:
        lbl = label(col)
        row = miss_df.loc[col]
        print(f"  {col:38s}  {lbl}")
        print(f"      Train: {int(row['Train Missings']):>7,}  ({row['Train %']:>5.1f}%)"
              f"   Test: {int(row['Test Missings']):>7,}  ({row['Test %']:>5.1f}%)")

    # Duplicates
    train_dup = train["front_id"].duplicated().sum()
    test_dup = test["front_id"].duplicated().sum()
    print(f"\n--- Duplicate front_id ---")
    print(f"Train: {train_dup}")
    print(f"Test:  {test_dup}")

    # Column classification
    id_col = "front_id"
    target_col = "target_value"
    categorical_cols = train.select_dtypes(include="object").columns.tolist()
    if id_col in categorical_cols:
        categorical_cols.remove(id_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    # Also treat db_group_last and fl_adminarea as categorical even if not object
    for col in ["db_group_last", "fl_adminarea"]:
        if col not in categorical_cols:
            categorical_cols.append(col)
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if id_col in numeric_cols:
        numeric_cols.remove(id_col)
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    print(f"\n--- Column summary ---")
    print(f"ID column:            {id_col}")
    print(f"Target column:        {target_col}  ({label(target_col)})")
    print(f"Numeric features:     {len(numeric_cols)}")
    for col in numeric_cols:
        print(f"    {col:40s} {label(col)}")
    print(f"Categorical features: {len(categorical_cols)}")
    for col in categorical_cols:
        print(f"    {col:40s} {label(col)}")

    result = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "train_dtypes": train.dtypes,
        "train_missing": miss_df,
        "test_missing": test_miss,
        "train_duplicates": train_dup,
        "test_duplicates": test_dup,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "id_col": id_col,
    }
    return result


# ===================================================================
# 2. Target variable analysis
# ===================================================================

def analyze_target(train: pd.DataFrame, target_col: str = "target_value") -> dict:
    """
    Analyze and plot target variable distribution.

    Returns dict with 'value_counts', 'proportions', 'class_balance_ratio'.
    """
    print("=" * 60)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 60)

    vc = train[target_col].value_counts().sort_index()
    prop = train[target_col].value_counts(normalize=True).sort_index()

    print(f"\nValue counts:\n{vc.to_string()}")
    print(f"\nProportions:\n{prop.to_string()}")
    print(f"\nClass balance ratio (majority / minority): {prop.max() / prop.min():.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Count bar
    ax = axes[0]
    colors = sns.color_palette("Set2", 2)
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="gray")
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vc.values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution (Counts)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Declined (0)", "Accepted (1)"])

    # Pie
    ax = axes[1]
    ax.pie(prop.values, labels=[f"Declined (0)\n{prop[0]:.1%}", f"Accepted (1)\n{prop[1]:.1%}"],
           colors=colors, autopct="", startangle=90, wedgeprops={"edgecolor": "gray"})
    ax.set_title("Class Distribution (Proportions)")

    plt.tight_layout()
    plt.show()

    return {
        "value_counts": vc.to_dict(),
        "proportions": prop.to_dict(),
        "class_balance_ratio": prop.max() / prop.min(),
    }


# ===================================================================
# 3. Numerical feature analysis
# ===================================================================

def analyze_numerical(train: pd.DataFrame, numeric_cols: list,
                      target_col: str = "target_value",
                      max_cols_per_fig: int = 25) -> pd.DataFrame:
    """
    Plot distributions (histograms) and boxplots for all numeric features.

    Returns a summary DataFrame with: mean, std, min, 25%, 50%, 75%, max,
    skewness, kurtosis, missing%, and number of outliers (> 3*IQR).
    """
    print("=" * 60)
    print("NUMERICAL FEATURE ANALYSIS")
    print("=" * 60)

    train_num = train[numeric_cols].copy()

    # ---- Summary statistics ----
    desc = train_num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["skew"] = train_num.skew()
    desc["kurtosis"] = train_num.kurtosis()
    desc["missing%"] = 100 * train_num.isnull().mean()
    # Outlier count (3*IQR rule)
    q1 = train_num.quantile(0.25)
    q3 = train_num.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (train_num < (q1 - 3 * iqr)) | (train_num > (q3 + 3 * iqr))
    desc["outliers"] = outlier_mask.sum()

    desc = desc.rename(columns={
        "1%": "p1", "5%": "p5", "50%": "p50", "95%": "p95", "99%": "p99",
    })
    display_cols = ["mean", "std", "min", "p1", "p5", "p25", "p50", "p75",
                    "p95", "p99", "max", "skew", "kurtosis", "missing%", "outliers"]
    available = [c for c in display_cols if c in desc.columns]
    print("\n--- Summary statistics ---")
    # Build display table with labelled index
    display_desc = desc[available].copy()
    display_desc.index = [f"{idx}  ({label(idx)})" for idx in display_desc.index]
    print(display_desc.to_string(float_format=lambda x: f"{x:.4f}" if abs(x) < 10000 else f"{x:.0f}"))

    # ---- Distribution plots (histograms) ----
    n_cols = min(max_cols_per_fig, len(numeric_cols))
    n_rows = int(np.ceil(n_cols / 5))
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols[:n_cols]):
        ax = axes[i]
        data = train_num[col].dropna()
        if data.nunique() < 3:
            ax.text(0.5, 0.5, "Constant / near-constant", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{col}\n{label(col)}", fontsize=8)
            continue
        ax.hist(data, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_title(f"{col}\n{label(col)}\n(skew={train_num[col].skew():.2f})", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numerical Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # ---- Boxplots ----
    n_cols_bp = min(20, len(numeric_cols))
    n_rows_bp = int(np.ceil(n_cols_bp / 5))
    fig, axes = plt.subplots(n_rows_bp, 5, figsize=(20, 3 * n_rows_bp))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols[:n_cols_bp]):
        ax = axes[i]
        data = train_num[col].dropna()
        if data.nunique() < 3:
            ax.text(0.5, 0.5, "Constant", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{col}\n{label(col)}", fontsize=8)
            continue
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightblue"),
                   medianprops=dict(color="red"))
        ax.set_title(f"{col}\n{label(col)}", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numerical Feature Boxplots", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    return desc[available]


# ===================================================================
# 4. Categorical variable analysis
# ===================================================================

def analyze_categorical(train: pd.DataFrame, cat_cols: list,
                        target_col: str = "target_value",
                        max_categories: int = 15) -> dict:
    """
    Analyze categorical variables: value counts and target rate per category.

    Returns a dict keyed by column name with DataFrames containing
    count, proportion, and target rate.
    """
    print("=" * 60)
    print("CATEGORICAL VARIABLE ANALYSIS")
    print("=" * 60)

    results = {}
    for col in cat_cols:
        print(f"\n--- {col}  ({label(col)}) ---")
        print(f"  Unique values: {train[col].nunique()}")

        # Value counts
        vc = train[col].value_counts(dropna=False)
        prop = train[col].value_counts(dropna=False, normalize=True)

        # Target rate per category
        target_rate = train.groupby(col, dropna=False)[target_col].mean()

        summary = pd.DataFrame({
            "count": vc,
            "proportion": prop,
            "target_rate": target_rate,
        }).sort_values("count", ascending=False)

        # Show top categories
        display = summary.head(max_categories)
        print(display.to_string(float_format=lambda x: f"{x:.4f}"))
        if len(summary) > max_categories:
            print(f"  ... and {len(summary) - max_categories} more categories")

        results[col] = summary

        # Plot top categories
        top15 = summary.head(15).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count bar
        ax = axes[0]
        palette = sns.color_palette("viridis", n_colors=min(15, len(top15)))
        bars = ax.barh(range(len(top15)), top15["count"], color=palette, edgecolor="gray")
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15.iloc[:, 0].astype(str), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title(f"Top Categories: {col}\n({label(col)})")

        # Target rate bar
        ax = axes[1]
        colors = ["#2ecc71" if v > 0.05 else "#e74c3c" for v in top15["target_rate"]]
        ax.barh(range(len(top15)), top15["target_rate"], color=colors, edgecolor="gray")
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15.iloc[:, 0].astype(str), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Target Rate")
        ax.set_title(f"Target Rate by {col}\n({label(col)})")
        ax.axvline(train[target_col].mean(), color="black", linestyle="--", linewidth=1, label="Global mean")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    return results


# ===================================================================
# 5. Correlation analysis
# ===================================================================

def analyze_correlations(train: pd.DataFrame, numeric_cols: list,
                         cat_cols: list, target_col: str = "target_value") -> dict:
    """
    Compute correlations with target:
      - Point-biserial correlation for numeric features
      - Cramér's V for categorical features

    Returns a dict with 'numeric_corr' (DataFrame) and 'categorical_corr' (DataFrame).
    """
    print("=" * 60)
    print("CORRELATION ANALYSIS WITH TARGET")
    print("=" * 60)

    # ---- Point-biserial (numeric vs binary target) ----
    print("\n--- Point-biserial correlation (numeric vs target) ---")
    pb_results = []
    for col in numeric_cols:
        valid = train[[col, target_col]].dropna()
        if len(valid) < 10:
            continue
        # Ensure both are numeric
        x = valid[col].values
        y = valid[target_col].values
        if np.ptp(x) == 0:  # constant
            continue
        coeff, pval = pointbiserialr(y, x)
        pb_results.append({"feature": col, "point_biserial_r": coeff, "p_value": pval})

    pb_df = pd.DataFrame(pb_results).sort_values("point_biserial_r", key=abs, ascending=False)
    pb_display = pb_df.copy()
    pb_display.insert(1, "description", pb_display["feature"].map(label))
    print(pb_display.to_string(float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    # ---- Cramér's V (categorical vs binary target) ----
    print("\n--- Cramér's V (categorical vs target) ---")
    cv_results = []
    for col in cat_cols:
        ct = pd.crosstab(train[col].fillna("__NaN__"), train[target_col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, _, _, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        k = ct.shape[0]  # rows
        r = ct.shape[1]  # cols
        cramers_v = np.sqrt(chi2 / (n * min(k - 1, r - 1)))
        cv_results.append({"feature": col, "cramers_v": cramers_v, "chi2": chi2})

    cv_df = pd.DataFrame(cv_results).sort_values("cramers_v", ascending=False)
    cv_display = cv_df.copy()
    cv_display.insert(1, "description", cv_display["feature"].map(label))
    print(cv_display.to_string(float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    # ---- Heatmap of numeric correlations ----
    plt.figure(figsize=(16, 13))
    corr_matrix = train[numeric_cols + [target_col]].corr()
    corr_short_labels = [label(c) for c in corr_matrix.columns]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    ax = sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="RdBu_r",
                     center=0, vmin=-1, vmax=1, square=True,
                     linewidths=0.3, cbar_kws={"shrink": 0.8})
    ax.set_xticklabels(corr_short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr_short_labels, rotation=0, fontsize=7)
    plt.title("Correlation Matrix (Numeric Features)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ---- Heatmap focused on target correlations ----
    target_corr = corr_matrix[[target_col]].drop(index=target_col).sort_values(
        target_col, ascending=False
    )
    target_labels = [label(c) for c in target_corr.index]
    plt.figure(figsize=(8, 0.35 * len(target_corr)))
    ax = sns.heatmap(target_corr.T, annot=True, cmap="RdBu_r", center=0,
                     vmin=-1, vmax=1, fmt=".3f", cbar_kws={"shrink": 0.5},
                     linewidths=0.5, yticklabels=False)
    ax.set_yticklabels([])
    ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=8)
    plt.title("Feature Correlations with Target", fontsize=12)
    plt.tight_layout()
    plt.show()

    return {
        "numeric_corr": pb_df,
        "categorical_corr": cv_df,
    }


# ===================================================================
# 6. Comprehensive EDA runner
# ===================================================================

def run_eda(train_path: str = "data/train_apps.csv",
            test_path: str = "data/test_apps.csv",
            target_col: str = "target_value") -> tuple:
    """
    Run the full EDA pipeline.

    Parameters
    ----------
    train_path : str
        Path to training CSV.
    test_path : str
        Path to test CSV.
    target_col : str
        Name of the target column.

    Returns
    -------
    (train, test, eda_results) where eda_results is a dict with all EDA outputs.
    """
    print("#" * 60)
    print("#  EXPLORATORY DATA ANALYSIS")
    print("#" * 60)

    # Load data
    print("\nLoading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")

    # 1. Overview
    overview_result = overview(train, test)

    # 2. Target
    target_result = analyze_target(train, target_col)

    numeric_cols = overview_result["numeric_cols"]
    categorical_cols = overview_result["categorical_cols"]

    # 3. Numeric
    num_summary = analyze_numerical(train, numeric_cols, target_col)

    # 4. Categorical
    cat_results = analyze_categorical(train, categorical_cols, target_col)

    # 5. Correlations
    corr_results = analyze_correlations(train, numeric_cols, categorical_cols, target_col)

    eda_results = {
        "overview": overview_result,
        "target": target_result,
        "numerical_summary": num_summary,
        "categorical": cat_results,
        "correlations": corr_results,
    }

    return train, test, eda_results


# ===================================================================
# Entry point for standalone execution
# ===================================================================
if __name__ == "__main__":
    train, test, eda_results = run_eda()
    print("\nEDA completed successfully.")
