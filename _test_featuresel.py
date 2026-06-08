import sys, os, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, os.getcwd())

from src.model.preprocessing import preprocess_data
from src.model.features import create_features
from src.model.models import train_models
from src.model.feature_selection import analyze_features

print('Loading data...')
train_df = pd.read_csv('data/train_apps.csv')
test_df = pd.read_csv('data/test_apps.csv')

X_tr, X_te, _ = preprocess_data(train_df, test_df)
X_tr, X_te, _ = create_features(X_tr, X_te, verbose=False)
y_tr = train_df['target_value']
print(f'X_tr: {X_tr.shape}')

print('\nTraining models (LightGBM + RF only, quick)...')
model_results = train_models(X_tr, y_tr, plot=False)
print(f'Best model: {model_results["best_model"]}')
print(f'Models trained: {list(model_results["results"].keys())}')

print('\nAnalysing features...')
analysis = analyze_features(model_results, X_tr, y_tr, plot=False)

print(f'\n=== Feature Selection Result ===')
print(f'Total:     {len(X_tr.columns)}')
print(f'Selected:  {len(analysis["selected_features"])}')
print(f'Dropped:   {len(analysis["dropped_features"])}')
if analysis['dropped_features']:
    print(f'Dropped:   {analysis["dropped_features"]}')
print()
print('Top 10 by combined importance:')
cols = ['feature', 'gain_importance', 'perm_importance_mean', 'shap_importance', 'avg_importance']
existing = [c for c in cols if c in analysis['importance_df'].columns]
print(analysis['importance_df'].head(10)[existing].to_string(float_format=lambda x: f'{x:.6f}'))
print()
print('OK!')
