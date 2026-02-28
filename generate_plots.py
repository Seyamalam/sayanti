import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set global aesthetics for Q1 paper
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

os.makedirs('manuscript', exist_ok=True)

# 1. Model Performance (Bar plot)
df_perf = pd.read_csv('results/model_performance.csv')
df_perf_melted = df_perf.melt(id_vars='model', value_vars=['auc_roc', 'auc_prc', 'recall'], var_name='Metric', value_name='Score')
df_perf_melted['Metric'] = df_perf_melted['Metric'].map({'auc_roc': 'AUC-ROC', 'auc_prc': 'AUC-PRC', 'recall': 'Recall'})
df_perf_melted['model'] = df_perf_melted['model'].map({'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'random_forest': 'Random Forest', 'logistic': 'Logistic Reg'})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='model', y='Score', hue='Metric', data=df_perf_melted, palette='viridis')
plt.ylim(0, 1.1)
plt.xlabel('Algorithm')
plt.ylabel('Score')
plt.legend(title='Metric', loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.tight_layout()
plt.savefig('manuscript/model_perf.pdf', dpi=300)
plt.close()

# 2. Out-of-Time Drift (Line plot)
df_cv = pd.read_csv('results/cross_validation.csv')
df_cv['model'] = df_cv['model'].map({'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'random_forest': 'Random Forest', 'logistic': 'Logistic Reg'})

plt.figure(figsize=(8, 5))
ax = sns.lineplot(x='fold', y='auc_roc', hue='model', style='model', markers=True, dashes=False, data=df_cv, linewidth=2.5, markersize=10, palette='Set1')
plt.xlabel('Chronological Fold (Time)')
plt.ylabel('AUC-ROC Score')
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0.5, 1.05)
plt.legend(title='Algorithm', loc='lower right')
plt.tight_layout()
plt.savefig('manuscript/oot_drift.pdf', dpi=300)
plt.close()

# 3. Feature Importance Comparison (Horizontal Bar Plot)
df_std_all = pd.read_csv('results/standard_shap_importance.csv')
df_ic_all = pd.read_csv('results/ic_shap_importance.csv')

top_features = list(set(df_std_all.head(5)['feature']).union(set(df_ic_all.head(5)['feature'])))

df_std_top = df_std_all[df_std_all['feature'].isin(top_features)].copy()
df_std_top['Method'] = 'Standard SHAP'
df_ic_top = df_ic_all[df_ic_all['feature'].isin(top_features)].copy()
df_ic_top['Method'] = 'IC-SHAP'

df_imp = pd.concat([df_std_top, df_ic_top])

# Sort by standard SHAP importance for consistent visual ordering
sort_order = df_std_top.sort_values(by='importance', ascending=False)['feature'].tolist()
df_imp['feature'] = pd.Categorical(df_imp['feature'], categories=sort_order, ordered=True)
df_imp = df_imp.sort_values('feature')

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', hue='Method', data=df_imp, palette='muted')
plt.xlabel('Mean Absolute SHAP Value (Importance)')
plt.ylabel('Feature')
plt.legend(title='Method', loc='lower right')
plt.tight_layout()
plt.savefig('manuscript/feat_imp.pdf', dpi=300)
plt.close()

# 4. Counterfactual Distribution (Histogram/Density of number of changes)
df_cf = pd.read_csv('results/counterfactuals.csv')
df_cf_success = df_cf[df_cf['success'] == True]

plt.figure(figsize=(8, 5))
sns.histplot(data=df_cf_success, x='n_changes', bins=15, kde=True, color='teal')
plt.xlabel('Number of Feature Modifications required for Recourse')
plt.ylabel('Frequency (Number of Cases)')
plt.axvline(x=df_cf_success['n_changes'].mean(), color='red', linestyle='--', label=f'Mean: {df_cf_success["n_changes"].mean():.1f} changes')
plt.legend()
plt.tight_layout()
plt.savefig('manuscript/cf_changes.pdf', dpi=300)
plt.close()
