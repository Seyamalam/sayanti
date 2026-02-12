"""
Explainable AI for Financial Fraud Detection Research - Optimized Version
==========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("XAI for Financial Fraud Detection Research")
print("="*70)


def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and preprocess data."""
    print(f"\n[1] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"    Dataset shape: {df.shape}, Fraud rate: {df['is_fraud'].mean():.4%}")
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['amt_log'] = np.log1p(df['amt'])
    
    df['dob'] = pd.to_datetime(df['dob'], format='%d-%m-%Y', errors='coerce')
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['age'] = df['age'].clip(18, 100)
    df['city_pop_log'] = np.log1p(df['city_pop'])
    
    target_enc = {}
    for col in ['merchant', 'job']:
        target_enc[col] = df.groupby(col)['is_fraud'].mean()
        df[f'{col}_enc'] = df[col].map(target_enc[col]).fillna(0)
    
    cat_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True)
    df = pd.concat([df, cat_dummies], axis=1)
    df['gender_enc'] = (df['gender'] == 'M').astype(int)
    
    feature_cols = [
        'amt', 'amt_log', 'hour', 'hour_sin', 'hour_cos', 'day_of_week',
        'is_weekend', 'distance', 'age', 'city_pop', 'city_pop_log',
        'merchant_enc', 'job_enc', 'gender_enc'
    ]
    feature_cols.extend([c for c in df.columns if c.startswith('cat_')])
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['is_fraud']
    
    print(f"    Processed: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_cols


def train_models(X_train, y_train, X_val, y_val):
    """Train baseline models."""
    models = {}
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("\n[2] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        max_depth=8, n_estimators=200, learning_rate=0.05,
        scale_pos_weight=scale, random_state=RANDOM_STATE,
        use_label_encoder=False, eval_metric='auc', n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models['xgboost'] = xgb_model
    
    print("[3] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        num_leaves=64, n_estimators=200, learning_rate=0.03,
        scale_pos_weight=scale, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    models['lightgbm'] = lgb_model
    
    print("[4] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        class_weight={0: 1, 1: scale}, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate models and return results."""
    results = []
    
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results.append({
            'model': name,
            'auc_roc': roc_auc_score(y_test, y_proba),
            'auc_prc': average_precision_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        })
    
    return pd.DataFrame(results)


def compute_shap_explanations(model, X_train, X_test, y_train, feature_names, n_samples=200):
    """Compute SHAP explanations (standard and IC-SHAP)."""
    print(f"\n[5] Computing SHAP explanations (sample={n_samples})...")
    
    X_sample = X_test[:n_samples]
    
    X_train_arr = np.asarray(X_train, dtype=np.float64)
    X_sample_arr = np.asarray(X_sample, dtype=np.float64)
    
    bg = X_train_arr[np.random.choice(len(X_train_arr), min(100, len(X_train_arr)), replace=False)]
    
    explainer = shap.TreeExplainer(model, data=bg, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_sample_arr)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    standard_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    X_legit = X_train_arr[y_train == 0]
    X_fraud = X_train_arr[y_train == 1]
    
    n_legit, n_fraud = len(X_legit), len(X_fraud)
    w_legit = n_fraud / (n_legit + n_fraud)
    w_fraud = n_legit / (n_legit + n_fraud)
    
    print(f"    IC-SHAP weights: w_legit={w_legit:.4f}, w_fraud={w_fraud:.4f}")
    
    bg_legit = X_legit[np.random.choice(len(X_legit), min(50, len(X_legit)), replace=False)]
    bg_fraud = X_fraud[np.random.choice(len(X_fraud), min(50, len(X_fraud)), replace=len(X_fraud) < 50)]
    
    exp_legit = shap.TreeExplainer(model, data=bg_legit, feature_perturbation='interventional')
    exp_fraud = shap.TreeExplainer(model, data=bg_fraud, feature_perturbation='interventional')
    
    shap_legit = exp_legit.shap_values(X_sample_arr)
    shap_fraud = exp_fraud.shap_values(X_sample_arr)
    
    if isinstance(shap_legit, list):
        shap_legit, shap_fraud = shap_legit[1], shap_fraud[1]
    
    ic_shap_values = w_legit * shap_legit + w_fraud * shap_fraud
    
    ic_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(ic_shap_values).mean(axis=0),
        'mean_shap': ic_shap_values.mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    return standard_importance, ic_importance, shap_values, ic_shap_values


def compute_counterfactuals(model, X_test, y_test, feature_names, n_samples=10):
    """Generate counterfactual explanations."""
    print(f"\n[6] Generating counterfactual explanations...")
    
    from scipy.optimize import minimize
    
    fraud_idx = np.where(y_test == 1)[0][:n_samples]
    if len(fraud_idx) == 0:
        return pd.DataFrame()
    
    bounds = [(X_test[:, i].min(), X_test[:, i].max()) for i in range(X_test.shape[1])]
    immutable = [feature_names.index(f) for f in ['age', 'gender_enc'] if f in feature_names]
    
    results = []
    
    for idx in fraud_idx:
        x_orig = X_test[idx]
        
        def objective(x_cf):
            try:
                pred = model.predict_proba(x_cf.reshape(1, -1))[0, 1]
            except:
                pred = 0.5
            pred_loss = max(0, pred - 0.5)
            dist_loss = np.linalg.norm(x_cf - x_orig)
            sparse_loss = np.sum(np.abs(x_cf - x_orig) > 1e-4)
            return 10 * pred_loss + 0.1 * dist_loss + 0.5 * sparse_loss
        
        constraints = [{'type': 'eq', 'fun': lambda x, i=i: x[i] - x_orig[i]} for i in immutable]
        
        result = minimize(objective, x_orig, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 500})
        
        x_cf = result.x
        changes = np.abs(x_cf - x_orig) > 1e-4
        
        results.append({
            'instance_idx': int(idx),
            'original_pred': float(model.predict_proba(x_orig.reshape(1, -1))[0, 1]),
            'cf_pred': float(model.predict_proba(x_cf.reshape(1, -1))[0, 1]),
            'n_changes': int(np.sum(changes)),
            'changed_features': ', '.join([feature_names[i] for i in range(len(changes)) if changes[i]][:5]),
            'success': result.success
        })
    
    return pd.DataFrame(results)


def evaluate_xai_quality(model, explainer, X_test, feature_names, n_samples=100):
    """Evaluate XAI quality metrics."""
    print(f"\n[7] Evaluating XAI quality...")
    
    X_sample = X_test[:n_samples]
    X_sample = np.asarray(X_sample, dtype=np.float64)
    
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[1] if len(base_val) > 1 else base_val[0]
    
    reconstructed = base_val + shap_values.sum(axis=1)
    actual = model.predict_proba(X_sample)[:, 1]
    fidelity = np.corrcoef(reconstructed, actual)[0, 1]
    
    base_shap = explainer.shap_values(X_sample[:1])
    if isinstance(base_shap, list):
        base_shap = base_shap[1][0]
    else:
        base_shap = base_shap[0]
    
    base_top = set(np.argsort(np.abs(base_shap))[-10:])
    jaccard_scores = []
    
    for _ in range(5):
        X_pert = X_sample[:1] + np.random.normal(0, 0.01, X_sample[:1].shape)
        pert_shap = explainer.shap_values(X_pert)
        if isinstance(pert_shap, list):
            pert_shap = pert_shap[1][0]
        else:
            pert_shap = pert_shap[0]
        pert_top = set(np.argsort(np.abs(pert_shap))[-10:])
        jaccard_scores.append(len(base_top & pert_top) / len(base_top | pert_top))
    
    stability = np.mean(jaccard_scores)
    
    abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(abs_shap)[::-1]
    cumulative = np.cumsum(abs_shap[sorted_idx]) / abs_shap.sum()
    comprehensibility = np.searchsorted(cumulative, 0.95) + 1
    
    return {
        'fidelity': float(fidelity),
        'stability': float(stability),
        'comprehensibility': int(comprehensibility)
    }


def generate_plots(model_results, std_imp, ic_imp, xai_quality, output_dir):
    """Generate and save visualizations."""
    print(f"\n[8] Generating visualizations...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models = model_results['model'].tolist()
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, model_results['auc_roc'], width, label='AUC-ROC', color='#3498db')
    axes[0, 0].bar(x + width/2, model_results['auc_prc'], width, label='AUC-PRC', color='#e74c3c')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.upper() for m in models])
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.1])
    
    top_n = 15
    axes[0, 1].barh(range(top_n), ic_imp['importance'].head(top_n).values, color='#2ecc71')
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(ic_imp['feature'].head(top_n).values)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel('Mean |SHAP|')
    axes[0, 1].set_title('IC-SHAP Feature Importance')
    
    top_idx = np.argsort(ic_imp['importance'].values)[-10:]
    std_vals = std_imp.set_index('feature').loc[ic_imp['feature'].iloc[top_idx]]['importance'].values
    ic_vals = ic_imp['importance'].iloc[top_idx].values
    
    axes[1, 0].scatter(std_vals, ic_vals, c='#9b59b6', s=100, alpha=0.7)
    axes[1, 0].plot([0, max(ic_vals)], [0, max(ic_vals)], 'k--', alpha=0.3)
    axes[1, 0].set_xlabel('Standard SHAP')
    axes[1, 0].set_ylabel('IC-SHAP')
    axes[1, 0].set_title('Standard vs IC-SHAP Comparison')
    
    metrics = ['fidelity', 'stability']
    values = [xai_quality[m] for m in metrics]
    bars = axes[1, 1].bar(metrics, values, color=['#1abc9c', '#f39c12'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('XAI Quality Metrics')
    axes[1, 1].set_ylim([0, 1.1])
    for bar, v in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/research_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_dir}/research_results.png")


def main():
    """Main experiment pipeline."""
    X, y, feature_names = load_and_preprocess('fraudTest.csv')
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X.values, y.values, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    print(f"\n    Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    models = train_models(X_train, y_train, X_val, y_val)
    
    model_results = evaluate_models(models, X_test, y_test)
    print("\n" + "="*70)
    print("MODEL RESULTS")
    print("="*70)
    print(model_results.to_string(index=False))
    model_results.to_csv(f'{OUTPUT_DIR}/model_performance.csv', index=False)
    
    xgb_model = models['xgboost']
    
    std_imp, ic_imp, std_shap, ic_shap = compute_shap_explanations(
        xgb_model, X_train, X_test, y_train, feature_names, n_samples=200
    )
    
    print("\n[IC-SHAP] Top 10 Features:")
    print(ic_imp.head(10).to_string(index=False))
    
    std_imp.to_csv(f'{OUTPUT_DIR}/standard_shap_importance.csv', index=False)
    ic_imp.to_csv(f'{OUTPUT_DIR}/ic_shap_importance.csv', index=False)
    
    explainer = shap.TreeExplainer(xgb_model)
    
    xai_quality = evaluate_xai_quality(xgb_model, explainer, X_test, feature_names)
    print(f"\n[XAI Quality] Fidelity: {xai_quality['fidelity']:.4f}, Stability: {xai_quality['stability']:.4f}")
    
    pd.DataFrame([xai_quality]).to_csv(f'{OUTPUT_DIR}/xai_quality.csv', index=False)
    
    cf_results = compute_counterfactuals(xgb_model, X_test, y_test, feature_names)
    if not cf_results.empty:
        print(f"\n[Counterfactuals] Generated for {len(cf_results)} fraud instances")
        cf_results.to_csv(f'{OUTPUT_DIR}/counterfactuals.csv', index=False)
    
    generate_plots(model_results, std_imp, ic_imp, xai_quality, OUTPUT_DIR)
    
    summary = {
        'dataset': {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'fraud_rate': float(y.mean()),
            'imbalance_ratio': float((1-y.mean())/y.mean())
        },
        'best_model': model_results.loc[model_results['auc_roc'].idxmax(), 'model'],
        'best_auc_roc': float(model_results['auc_roc'].max()),
        'xai_quality': xai_quality
    }
    
    with open(f'{OUTPUT_DIR}/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE - Results saved to ./results/")
    print("="*70)
    print(f"\nFiles saved:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {OUTPUT_DIR}/{f}")
    
    return summary


if __name__ == "__main__":
    main()
