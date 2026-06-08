import sys, os, pandas as pd, numpy as np, matplotlib, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

from model.preprocessing import preprocess_data
from model.features import create_features
from model.models import train_models
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ── Load ──────────────────────────────────────────────────────────────
train_df = pd.read_csv('data/train_apps.csv')
test_df = pd.read_csv('data/test_apps.csv')

# ── 1. Temporal check ─────────────────────────────────────────────────
print('=' * 60)
print('1. TEMPORAL SHIFT ANALYSIS')
print('=' * 60)
train_dates = pd.to_datetime(train_df['decision_day'])
test_dates = pd.to_datetime(test_df['decision_day'])
print(f'Train date range: {train_dates.min().date()} to {train_dates.max().date()}')
print(f'Test date range:  {test_dates.min().date()} to {test_dates.max().date()}')
print(f'Train/test overlap: {set(train_dates.dt.date) & set(test_dates.dt.date)}')
print(f'Unique dates: train={train_dates.nunique()}, test={test_dates.nunique()}')
# Train last 10% vs first 90% target rate
train_sorted = train_df.sort_values('decision_day')
n = len(train_sorted)
train_early = train_sorted.iloc[:int(n*0.9)]
train_late = train_sorted.iloc[int(n*0.9):]
print(f'Early train (first 90%) target rate: {train_early["target_value"].mean():.4f}')
print(f'Late train  (last 10%)  target rate: {train_late["target_value"].mean():.4f}')

# ── 2. Baseline check ─────────────────────────────────────────────────
print()
print('=' * 60)
print('2. BASELINE ACCURACY CHECK')
print('=' * 60)
print(f'Train target rate: {train_df["target_value"].mean():.4f}')
print(f'Always-0 accuracy on test: assumes ~{1 - train_df["target_value"].mean():.2%}')
print(f'User reported accuracy: 66%')
print(f'This implies the model is WORSE than always predicting 0')

# ── 3. Preprocess and get predictions ────────────────────────────────
print()
print('=' * 60)
print('3. PROCESSING FULL PIPELINE')
print('=' * 60)

X_tr, X_te, prep = preprocess_data(train_df, test_df)
X_tr_aug, X_te_aug, _ = create_features(X_tr, X_te, verbose=False)
y_tr = train_df['target_value']

# Train LightGBM with default params
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# 5-fold CV OOF
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.empty(len(y_tr))
test_preds = np.zeros((len(X_te_aug), 5))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_aug, y_tr)):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7,
                           num_leaves=63, class_weight='balanced',
                           random_state=42, verbosity=-1, n_jobs=-1)
    m.fit(X_tr_aug.iloc[tr_idx], y_tr.iloc[tr_idx])
    oof_preds[val_idx] = m.predict_proba(X_tr_aug.iloc[val_idx])[:, 1]
    test_preds[:, fold] = m.predict_proba(X_te_aug)[:, 1]

test_proba = test_preds.mean(axis=1)

print(f'Train OOF ROC-AUC: {roc_auc_score(y_tr, oof_preds):.4f}')
print(f'Test proba range:  {test_proba.min():.4f} - {test_proba.max():.4f}')
print(f'Train OOF proba range: {oof_preds.min():.4f} - {oof_preds.max():.4f}')

# ── 4. Probability distribution comparison ───────────────────────────
print()
print('=' * 60)
print('4. PROBABILITY DISTRIBUTIONS')
print('=' * 60)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
for label, color in [(0, 'steelblue'), (1, 'crimson')]:
    mask = y_tr == label
    ax.hist(oof_preds[mask], bins=50, alpha=0.6, color=color,
            label=f'Train class {label} (n={mask.sum()})', density=True)
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Density')
ax.set_title('Train OOF probabilities by true class')
ax.legend()

ax = axes[1]
ax.hist(test_proba, bins=50, color='green', alpha=0.6, label='Test predictions')
ax.axvline(test_proba.mean(), color='darkgreen', ls='--', label=f"Mean={test_proba.mean():.3f}")
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Count')
ax.set_title(f'Test probabilities (n={len(test_proba)})')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/probability_distributions.png', dpi=100)
print('  Saved: outputs/probability_distributions.png')

# ── 5. Threshold sweep ───────────────────────────────────────────────
print()
print('=' * 60)
print('5. THRESHOLD SWEEP (on train OOF)')
print('=' * 60)
thresholds = np.linspace(0.01, 0.99, 200)
results = []
for t in thresholds:
    yp = (oof_preds >= t).astype(int)
    results.append({
        'threshold': t,
        'accuracy': accuracy_score(y_tr, yp),
        'f1': f1_score(y_tr, yp, zero_division=0),
        'pred_rate': yp.mean(),
    })
res_df = pd.DataFrame(results)
best_acc_idx = res_df['accuracy'].idxmax()
best_f1_idx = res_df['f1'].idxmax()
print(f'Best accuracy threshold: {res_df.iloc[best_acc_idx]["threshold"]:.4f}  '
      f'(acc={res_df.iloc[best_acc_idx]["accuracy"]:.4f}, '
      f'pred_rate={res_df.iloc[best_acc_idx]["pred_rate"]:.4f})')
print(f'Best F1 threshold:      {res_df.iloc[best_f1_idx]["threshold"]:.4f}  '
      f'(acc={res_df.iloc[best_f1_idx]["accuracy"]:.4f}, '
      f'pred_rate={res_df.iloc[best_f1_idx]["pred_rate"]:.4f})')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(res_df['threshold'], res_df['accuracy'], label='Accuracy', linewidth=2)
ax.plot(res_df['threshold'], res_df['f1'], label='F1', linewidth=2)
ax.axvline(res_df.iloc[best_f1_idx]['threshold'], color='red', ls='--', alpha=0.5,
           label=f"Best F1 t={res_df.iloc[best_f1_idx]['threshold']:.3f}")
ax.axvline(res_df.iloc[best_acc_idx]['threshold'], color='blue', ls='--', alpha=0.5,
           label=f"Best Acc t={res_df.iloc[best_acc_idx]['threshold']:.3f}")
ax.axhline(train_df['target_value'].mean()*0 + (1-train_df['target_value'].mean()),
           color='gray', ls=':', label='Always-0 baseline')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold sweep (train OOF)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/threshold_sweep.png', dpi=100)
print('  Saved: outputs/threshold_sweep.png')

# What if we use accuracy-tuned threshold on test?
best_acc_t = res_df.iloc[best_acc_idx]['threshold']
best_f1_t = res_df.iloc[best_f1_idx]['threshold']
print()
print(f'If we used acc-optimised threshold ({best_acc_t:.4f}):')
print(f'  Test pred rate: {(test_proba >= best_acc_t).mean():.4f}')
print(f'If we used F1-optimised threshold ({best_f1_t:.4f}):')
print(f'  Test pred rate: {(test_proba >= best_f1_t).mean():.4f}')

# ── 6. Feature distribution drift ────────────────────────────────────
print()
print('=' * 60)
print('6. FEATURE DISTRIBUTION DRIFT (top 10 by importance)')
print('=' * 60)
important_cols = [
    'loan_amount_last', 'cb_rate', 'overdraft_limit_max', 'sum_deb_ul_90',
    'count_all_corp_dashboard_events', 'days_from_authperson_registration',
    'overdraft_limit_min', 'fl_adminarea_te', 'rate_spread', 'db_group_last_te'
]
for col in important_cols:
    if col in X_tr_aug.columns and col in X_te_aug.columns:
        tr_mean = X_tr_aug[col].mean()
        te_mean = X_te_aug[col].mean()
        tr_std = X_tr_aug[col].std()
        te_std = X_te_aug[col].std()
        shift = abs(te_mean - tr_mean) / (tr_std + 1e-8)
        print(f'  {col:40s}  train={tr_mean:.4f}±{tr_std:.4f}  test={te_mean:.4f}±{te_std:.4f}  shift={shift:.3f}σ')

# ── 7. Engineered feature extremes ───────────────────────────────────
print()
print('=' * 60)
print('7. ENGINEERED FEATURE EXTREMES')
print('=' * 60)
for col in ['rate_spread', 'limit_request_ratio', 'limit_range',
            'activity_ratio_30_90', 'log_balance', 'activity_decay_90_30',
            'sum_deb_total', 'payment_to_income', 'loan_amount_ratio']:
    if col in X_tr_aug.columns and col in X_te_aug.columns:
        tr99 = X_tr_aug[col].quantile(0.99)
        te99 = X_te_aug[col].quantile(0.99)
        tr01 = X_tr_aug[col].quantile(0.01)
        te01 = X_te_aug[col].quantile(0.01)
        print(f'  {col:25s}  train p1={tr01:>10.4f} p99={tr99:>10.4f}  '
              f'test p1={te01:>10.4f} p99={te99:>10.4f}')

print()
print('Diagnostics complete.')
