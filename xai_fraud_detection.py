"""
XAI Fraud Detection Research Framework - Full Version
=====================================================
Complete implementation with extensive logging for debugging.

Run: python xai_fraud_detection.py --data fraudTest.csv --output results
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
import shap

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
    print("[LOG] LIME package available")
except ImportError:
    LIME_AVAILABLE = False
    print("[LOG] LIME not available - will skip LIME experiments")

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("XAI FRAUD DETECTION RESEARCH FRAMEWORK - STARTING")
print("="*80)
print(f"[LOG] Random seed: {RANDOM_STATE}")
print(f"[LOG] Timestamp: {datetime.now().isoformat()}")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    RANDOM_STATE = 42
    N_FOLDS = 5
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    N_BACKGROUND_SAMPLES = 100
    N_EXPLANATION_SAMPLES = 200
    N_COUNTERFACTUAL_SAMPLES = 50  # Increased for better statistics
    TARGET_FIDELITY = 0.90
    TARGET_STABILITY = 0.85
    TARGET_COMPREHENSIBILITY = 10
    EPSILON_FEATURE_CHANGE = 1e-4

print("[LOG] Configuration loaded")
print(f"  - N_FOLDS: {Config.N_FOLDS}")
print(f"  - N_COUNTERFACTUAL_SAMPLES: {Config.N_COUNTERFACTUAL_SAMPLES}")
print(f"  - N_EXPLANATION_SAMPLES: {Config.N_EXPLANATION_SAMPLES}")


# =============================================================================
# DATA PREPROCESSING WITH LOGGING
# =============================================================================

print("\n" + "="*80)
print("PHASE 1: DATA PREPROCESSING")
print("="*80)

class DataPreprocessor:
    def __init__(self, random_state: int = RANDOM_STATE):
        print("[LOG] DataPreprocessor initialized")
        self.random_state = random_state
        self.feature_names = []
        self.target_encodings = {}
        self.fitted = False
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        print(f"\n[LOG] Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"[LOG] Data loaded successfully")
        print(f"  - Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  - Fraud rate: {df['is_fraud'].mean():.4%}")
        print(f"  - Fraud count: {df['is_fraud'].sum():,}")
        print(f"  - Legitimate count: {(df['is_fraud']==0).sum():,}")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[LOG] Engineering features...")
        df = df.copy()
        
        # Temporal
        print("  - Extracting temporal features...")
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['month'] = df['trans_date_trans_time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Geographic
        print("  - Computing geographic features...")
        df['distance'] = self._haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
        df['lat_diff'] = np.abs(df['lat'] - df['merch_lat'])
        df['long_diff'] = np.abs(df['long'] - df['merch_long'])
        
        # Amount
        print("  - Transforming amount features...")
        df['amt_log'] = np.log1p(df['amt'])
        df['amt_sqrt'] = np.sqrt(df['amt'])
        
        # Demographics
        print("  - Computing demographic features...")
        df['dob'] = pd.to_datetime(df['dob'], format='%d-%m-%Y', errors='coerce')
        df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days // 365).clip(18, 100)
        df['city_pop_log'] = np.log1p(df['city_pop'])
        df['gender_enc'] = (df['gender'] == 'M').astype(int)
        
        # Categories
        print("  - One-hot encoding categories...")
        cat_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True)
        df = pd.concat([df, cat_dummies], axis=1)
        
        print(f"[LOG] Feature engineering complete. Total columns: {df.shape[1]}")
        return df
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    def fit_transform(self, df: pd.DataFrame, fit_encodings: bool = True):
        print("\n[LOG] Fitting preprocessor and transforming data...")
        
        # IMPORTANT: Target encoding computed ONLY on training data
        if fit_encodings:
            print("  [IMPORTANT] Computing target encodings on TRAINING data only (no leakage)")
            for col in ['merchant', 'job']:
                self.target_encodings[col] = df.groupby(col)['is_fraud'].mean()
                print(f"    - {col}: {len(self.target_encodings[col])} unique values encoded")
        
        # Apply target encoding
        print("  - Applying target encodings...")
        for col in ['merchant', 'job']:
            df[f'{col}_enc'] = df[col].map(self.target_encodings.get(col, {})).fillna(0)
        
        # Feature list
        self.feature_names = [
            'amt', 'amt_log', 'hour', 'hour_sin', 'hour_cos',
            'day_of_week', 'is_weekend', 'month', 'distance',
            'age', 'city_pop_log', 'merchant_enc', 'job_enc', 'gender_enc'
        ]
        self.feature_names += [c for c in df.columns if c.startswith('cat_')]
        
        X = df[self.feature_names].fillna(0).values.astype(np.float64)
        y = df['is_fraud'].values
        
        print(f"[LOG] Preprocessing complete:")
        print(f"  - Features: {len(self.feature_names)}")
        print(f"  - Samples: {X.shape[0]:,}")
        
        return X, y, self.feature_names


# =============================================================================
# MODEL TRAINING WITH LOGGING
# =============================================================================

print("\n" + "="*80)
print("PHASE 2: MODEL TRAINING")
print("="*80)

class ModelTrainer:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.scaler = None
    
    def train_all(self, X_train, y_train, X_val, y_val):
        scale = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        print(f"\n[LOG] Training all models with class weight: {scale:.2f}")
        print(f"  - Training samples: {X_train.shape[0]:,}")
        print(f"  - Validation samples: {X_val.shape[0]:,}")
        
        # XGBoost
        print("\n[LOG] Training XGBoost...")
        self.models['xgboost'] = xgb.XGBClassifier(
            max_depth=8, n_estimators=200, learning_rate=0.05,
            scale_pos_weight=scale, random_state=self.random_state,
            use_label_encoder=False, eval_metric='auc', n_jobs=-1
        )
        self.models['xgboost'].fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("  - XGBoost training complete")
        
        # LightGBM
        print("[LOG] Training LightGBM...")
        self.models['lightgbm'] = lgb.LGBMClassifier(
            num_leaves=64, n_estimators=200, learning_rate=0.03,
            scale_pos_weight=scale, random_state=self.random_state,
            verbose=-1, n_jobs=-1
        )
        self.models['lightgbm'].fit(X_train, y_train, eval_set=[(X_val, y_val)],
                                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        print("  - LightGBM training complete")
        
        # Random Forest
        print("[LOG] Training Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight={0: 1, 1: scale},
            random_state=self.random_state, n_jobs=-1
        )
        self.models['random_forest'].fit(X_train, y_train)
        print("  - Random Forest training complete")
        
        # Logistic Regression
        print("[LOG] Training Logistic Regression...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.models['logistic'] = LogisticRegression(
            C=0.1, class_weight={0: 1, 1: scale},
            solver='saga', max_iter=500, random_state=self.random_state
        )
        self.models['logistic'].fit(X_train_scaled, y_train)
        print("  - Logistic Regression training complete")
        
        print("\n[LOG] All models trained successfully")
        return self.models
    
    def evaluate(self, X_test, y_test, feature_names):
        print("\n[LOG] Evaluating models...")
        results = []
        
        for name, model in self.models.items():
            X_t = self.scaler.transform(X_test) if name == 'logistic' else X_test
            y_proba = model.predict_proba(X_t)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Detection rate at different thresholds
            sorted_idx = np.argsort(y_proba)[::-1]
            sorted_y = y_test[sorted_idx]
            n_fraud = y_test.sum()
            
            det_rates = {}
            for pct in [0.01, 0.05, 0.10]:
                n_top = int(len(y_test) * pct)
                if n_top > 0 and n_fraud > 0:
                    det_rates[f'detection_rate_{int(pct*100)}pct'] = sorted_y[:n_top].sum() / n_fraud
            
            result = {
                'model': name,
                'auc_roc': roc_auc_score(y_test, y_proba),
                'auc_prc': average_precision_score(y_test, y_proba),
                'f1_score': f1_score(y_test, y_pred),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
                **det_rates
            }
            results.append(result)
            print(f"  - {name}: AUC-ROC={result['auc_roc']:.4f}, Recall={result['recall']:.4f}")
        
        return pd.DataFrame(results)


# =============================================================================
# IC-SHAP WITH LOGGING
# =============================================================================

print("\n" + "="*80)
print("PHASE 3: IC-SHAP IMPLEMENTATION")
print("="*80)

class ImbalanceCalibratedSHAP:
    def __init__(self, model, X_train, y_train, n_background=Config.N_BACKGROUND_SAMPLES):
        print("[LOG] Initializing IC-SHAP...")
        self.model = model
        self.n_background = n_background
        
        self.X_legit = X_train[y_train == 0]
        self.X_fraud = X_train[y_train == 1]
        
        n_legit, n_fraud = len(self.X_legit), len(self.X_fraud)
        self.w_legit = 0.5
        self.w_fraud = 0.5
        
        print(f"[LOG] IC-SHAP calibration weights:")
        print(f"  - w_legit = {self.w_legit:.6f} (for majority background)")
        print(f"  - w_fraud = {self.w_fraud:.6f} (for minority background)")
        print(f"  - Legitimate samples: {n_legit:,}")
        print(f"  - Fraud samples: {n_fraud:,}")
    
    def explain(self, X, feature_names):
        print("\n[LOG] Computing IC-SHAP explanations...")
        
        # Stratified sampling
        print("  - Creating stratified background samples...")
        bg_legit = self.X_legit[
            np.random.choice(len(self.X_legit), min(self.n_background, len(self.X_legit)), replace=False)
        ]
        bg_fraud = self.X_fraud[
            np.random.choice(len(self.X_fraud), min(self.n_background, len(self.X_fraud)),
                           replace=len(self.X_fraud) < self.n_background)
        ]
        print(f"    - Background legit: {len(bg_legit)}")
        print(f"    - Background fraud: {len(bg_fraud)}")
        
        # Class-conditional SHAP
        print("  - Computing class-conditional SHAP values...")
        try:
            exp_legit = shap.TreeExplainer(self.model, bg_legit)
            exp_fraud = shap.TreeExplainer(self.model, bg_fraud)
            shap_legit = exp_legit.shap_values(X)
            shap_fraud = exp_fraud.shap_values(X)
            if isinstance(shap_legit, list):
                shap_legit, shap_fraud = shap_legit[1], shap_fraud[1]
            print("    - TreeExplainer successful")
        except Exception as e:
            print(f"    - TreeExplainer failed: {e}")
            print("    - Falling back to standard SHAP")
            exp = shap.TreeExplainer(self.model)
            shap_legit = exp.shap_values(X)
            shap_fraud = shap_legit.copy()
            if isinstance(shap_legit, list):
                shap_legit, shap_fraud = shap_legit[1], shap_fraud[1]
        
        # Weighted aggregation
        print("  - Applying weighted aggregation...")
        ic_shap_values = self.w_legit * shap_legit + self.w_fraud * shap_fraud
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(ic_shap_values).mean(axis=0),
            'mean_shap': ic_shap_values.mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        print(f"[LOG] IC-SHAP complete. Top feature: {importance.iloc[0]['feature']}")
        return ic_shap_values, importance


# =============================================================================
# LIME EXPLAINER
# =============================================================================

print("\n" + "="*80)
print("PHASE 4: LIME BASELINE")
print("="*80)

def run_lime_analysis(model, X_train, X_sample, feature_names):
    if not LIME_AVAILABLE:
        print("[LOG] LIME not available - skipping")
        return None, None
    
    print("[LOG] Running LIME analysis...")
    print(f"  - Creating LIME explainer with {X_train.shape[0]:,} training samples")
    
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Legitimate', 'Fraud'],
        mode='classification'
    )
    
    print(f"  - Explaining {X_sample.shape[0]} samples...")
    all_importance = np.zeros((len(X_sample), len(feature_names)))
    
    for i, x in enumerate(X_sample):
        if i % 50 == 0:
            print(f"    - Progress: {i}/{len(X_sample)}")
        exp = explainer.explain_instance(x, model.predict_proba, num_features=len(feature_names))
        for feat_idx, weight in exp.local_exp[1]:
            all_importance[i, feat_idx] = weight
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(all_importance).mean(axis=0),
        'mean_weight': all_importance.mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print(f"[LOG] LIME complete. Top feature: {importance.iloc[0]['feature']}")
    return all_importance, importance


# =============================================================================
# COUNTERFACTUAL GENERATOR
# =============================================================================

print("\n" + "="*80)
print("PHASE 5: COUNTERFACTUAL GENERATION")
print("="*80)

class CounterfactualGenerator:
    def __init__(self, model, feature_names, immutable_features=None):
        self.model = model
        self.feature_names = feature_names
        self.immutable_features = immutable_features or ['age', 'gender_enc']
        self.immutable_indices = [
            feature_names.index(f) for f in self.immutable_features if f in feature_names
        ]
        print(f"[LOG] CounterfactualGenerator initialized")
        print(f"  - Immutable features: {self.immutable_features}")
        print(f"  - Immutable indices: {self.immutable_indices}")
    
    def generate(self, x_original, threshold=0.5, max_iterations=500, n_candidates=5):
        bounds = [(None, None) for _ in range(len(x_original))]
        
        # Helper to get feature index
        def idx(name):
            return self.feature_names.index(name)
            
        constraints = [
            {'type': 'eq', 'fun': lambda x, i=i: x[i] - x_original[i]}
            for i in self.immutable_indices
        ]
        
        # Structural constraints to prevent impossible counterfactuals
        if 'hour' in self.feature_names and 'hour_sin' in self.feature_names and 'hour_cos' in self.feature_names:
            constraints.append({'type': 'eq', 'fun': lambda x: x[idx('hour_sin')] - np.sin(2 * np.pi * x[idx('hour')] / 24)})
            constraints.append({'type': 'eq', 'fun': lambda x: x[idx('hour_cos')] - np.cos(2 * np.pi * x[idx('hour')] / 24)})
            
        if 'amt' in self.feature_names and 'amt_log' in self.feature_names:
            constraints.append({'type': 'eq', 'fun': lambda x: x[idx('amt_log')] - np.log1p(max(0, x[idx('amt')]))})
        
        def objective(x_cf):
            try:
                pred_margin = self.model.predict(x_cf.reshape(1, -1), output_margin=True)[0]
                pred = 1 / (1 + np.exp(-pred_margin))
            except:
                pred = 0.5
            pred_loss = max(0, pred - (1 - threshold))
            dist_loss = np.linalg.norm(x_cf - x_original)
            # Fix: L1 norm instead of step function so optimizer sees the gradient
            sparse_loss = np.sum(np.abs(x_cf - x_original))
            return 10 * pred_loss + 0.1 * dist_loss + 0.5 * sparse_loss
        
        best_result = None
        best_n_changes = float('inf')
        
        for seed in range(n_candidates):
            np.random.seed(seed)
            x_init = x_original.copy()
            for i in range(len(x_init)):
                if i not in self.immutable_indices:
                    x_init[i] += np.random.normal(0, 0.01)
            
            result = minimize(objective, x_init, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': max_iterations, 'ftol': 1e-6})
            
            x_cf = result.x
            changes = np.abs(x_cf - x_original) > Config.EPSILON_FEATURE_CHANGE
            n_changes = np.sum(changes)
            
            try:
                cf_pred = self.model.predict_proba(x_cf.reshape(1, -1))[0, 1]
            except:
                cf_pred = 0.5
            
            if cf_pred < threshold and n_changes < best_n_changes:
                best_n_changes = n_changes
                best_result = {
                    'success': True,
                    'counterfactual': x_cf,
                    'n_changes': int(n_changes),
                    'changed_features': [self.feature_names[i] for i in range(len(changes)) if changes[i]],
                    'original_pred': float(self.model.predict_proba(x_original.reshape(1, -1))[0, 1]),
                    'cf_pred': float(cf_pred)
                }
        
        if best_result is None:
            orig_pred = float(self.model.predict_proba(x_original.reshape(1, -1))[0, 1])
            best_result = {
                'success': False,
                'counterfactual': x_original,
                'n_changes': 0,
                'changed_features': [],
                'original_pred': orig_pred,
                'cf_pred': orig_pred
            }
        
        return best_result
    
    def batch_generate(self, X, y, n_samples=Config.N_COUNTERFACTUAL_SAMPLES):
        fraud_idx = np.where(y == 1)[0][:n_samples]
        
        if len(fraud_idx) == 0:
            print("[LOG] No fraud instances found")
            return pd.DataFrame(), {}
        
        print(f"\n[LOG] Generating counterfactuals for {len(fraud_idx)} fraud instances...")
        
        results = []
        successful = 0
        total_changes = []
        confidence_breakdown = {'high': 0, 'medium': 0, 'low': 0}
        
        for i, idx in enumerate(fraud_idx):
            if i % 10 == 0:
                print(f"  - Progress: {i}/{len(fraud_idx)}")
            
            cf = self.generate(X[idx])
            cf['instance_idx'] = int(idx)
            results.append(cf)
            
            if cf['success']:
                successful += 1
                total_changes.append(cf['n_changes'])
            
            # Track by confidence level
            if cf['original_pred'] > 0.99:
                confidence_breakdown['high'] += 1
            elif cf['original_pred'] > 0.95:
                confidence_breakdown['medium'] += 1
            else:
                confidence_breakdown['low'] += 1
        
        success_rate = successful / len(fraud_idx) * 100
        avg_changes = np.mean(total_changes) if total_changes else 0
        
        print(f"\n[LOG] Counterfactual generation complete:")
        print(f"  - Success rate: {success_rate:.1f}% ({successful}/{len(fraud_idx)})")
        print(f"  - Average feature changes: {avg_changes:.2f}")
        print(f"  - Confidence breakdown:")
        print(f"    - High (>0.99): {confidence_breakdown['high']}")
        print(f"    - Medium (0.95-0.99): {confidence_breakdown['medium']}")
        print(f"    - Low (<0.95): {confidence_breakdown['low']}")
        
        df_results = []
        for r in results:
            df_results.append({
                'instance_idx': r['instance_idx'],
                'original_pred': r['original_pred'],
                'cf_pred': r['cf_pred'],
                'n_changes': r['n_changes'],
                'changed_features': ', '.join(r['changed_features'][:5]),
                'success': r['success']
            })
        
        stats = {
            'success_rate': success_rate,
            'avg_changes': avg_changes,
            'n_successful': successful,
            'n_total': len(fraud_idx),
            'confidence_breakdown': confidence_breakdown
        }
        
        return pd.DataFrame(df_results), stats


# =============================================================================
# XAI QUALITY AUDITOR
# =============================================================================

print("\n" + "="*80)
print("PHASE 6: XAI QUALITY AUDIT")
print("="*80)

class ExplanationQualityAuditor:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        print("[LOG] ExplanationQualityAuditor initialized")
    
    def evaluate_fidelity(self, shap_values, base_value, X):
        print("[LOG] Evaluating fidelity...")
        reconstructed_margin = base_value + shap_values.sum(axis=1)
        # Fix: Need to convert sum of SHAP values (log-odds) back to probability space 
        # using sigmoid function before computing correlation
        reconstructed_prob = 1 / (1 + np.exp(-np.clip(reconstructed_margin, -15, 15)))
        actual_prob = self.model.predict_proba(X)[:, 1]
        correlation = np.corrcoef(reconstructed_prob, actual_prob)[0, 1]
        fidelity = float(correlation) if not np.isnan(correlation) else 0.0
        print(f"  - Fidelity: {fidelity:.4f}")
        return fidelity
    
    def evaluate_stability(self, explainer, X, n_perturbations=5):
        print("[LOG] Evaluating stability...")
        base_shap = explainer.shap_values(X[:1])
        if isinstance(base_shap, list):
            base_shap = base_shap[1][0]
        else:
            base_shap = base_shap[0]
        
        base_top = set(np.argsort(np.abs(base_shap))[-10:])
        jaccard_scores = []
        
        for _ in range(n_perturbations):
            X_pert = X[:1] + np.random.normal(0, 0.01, X[:1].shape)
            pert_shap = explainer.shap_values(X_pert)
            if isinstance(pert_shap, list):
                pert_shap = pert_shap[1][0]
            else:
                pert_shap = pert_shap[0]
            pert_top = set(np.argsort(np.abs(pert_shap))[-10:])
            jaccard = len(base_top & pert_top) / len(base_top | pert_top)
            jaccard_scores.append(jaccard)
        
        stability = float(np.mean(jaccard_scores))
        print(f"  - Stability: {stability:.4f}")
        return stability
    
    def evaluate_comprehensibility(self, shap_values, threshold=0.95):
        abs_importance = np.abs(shap_values).mean(axis=0)
        total = abs_importance.sum()
        if total == 0:
            return len(self.feature_names)
        sorted_idx = np.argsort(abs_importance)[::-1]
        cumulative = np.cumsum(abs_importance[sorted_idx]) / total
        comprehensibility = int(np.searchsorted(cumulative, threshold) + 1)
        print(f"  - Comprehensibility: {comprehensibility} features for 95% variance")
        return comprehensibility
    
    def audit(self, explainer, X, name="Explainer"):
        print(f"\n[LOG] Auditing {name}...")
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        fidelity = self.evaluate_fidelity(shap_values, base_value, X)
        stability = self.evaluate_stability(explainer, X)
        comprehensibility = self.evaluate_comprehensibility(shap_values)
        
        return {
            'fidelity': fidelity,
            'stability': stability,
            'comprehensibility': comprehensibility,
            'fidelity_met': fidelity >= Config.TARGET_FIDELITY,
            'stability_met': stability >= Config.TARGET_STABILITY
        }


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cross_validation(df, preprocessor, n_splits=Config.N_FOLDS):
    print(f"\n[LOG] Running {n_splits}-fold TimeSeriesSplit cross-validation...")
    # Fix: Sort chronologically to prevent temporal data leakage 
    if 'trans_date_trans_time' in df.columns:
        df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        print(f"\n[LOG] Fold {fold + 1}/{n_splits}")
        
        train_df_full = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        train_df, val_df = train_test_split(
            train_df_full, test_size=Config.VAL_SIZE,
            stratify=train_df_full['is_fraud'], random_state=RANDOM_STATE
        )
        
        # Fit-transform independently over time-splits to prevent target leakage
        X_train, y_train, feature_names = preprocessor.fit_transform(train_df, fit_encodings=True)
        X_val, y_val, _ = preprocessor.fit_transform(val_df, fit_encodings=False)
        X_test, y_test, _ = preprocessor.fit_transform(test_df, fit_encodings=False)
        
        trainer = ModelTrainer()
        trainer.train_all(X_train, y_train, X_val, y_val)
        
        fold_results = trainer.evaluate(X_test, y_test, feature_names)
        fold_results['fold'] = fold + 1
        results.append(fold_results)
    
    return pd.concat(results, ignore_index=True)


# =============================================================================
# ABLATION STUDY
# =============================================================================

def run_ablation_study(model, X_train, y_train, X_test, feature_names):
    print("\n[LOG] Running ablation study...")
    X_sample = X_test[:100]
    results = []
    
    # Standard SHAP
    print("  [1/4] Standard SHAP...")
    exp_std = shap.TreeExplainer(model)
    shap_std = exp_std.shap_values(X_sample)
    if isinstance(shap_std, list):
        shap_std = shap_std[1]
    std_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_std).mean(axis=0)
    }).sort_values('importance', ascending=False)
    results.append({
        'method': 'Standard SHAP',
        'top_feature': std_imp.iloc[0]['feature'],
        'mean_importance': std_imp['importance'].mean()
    })
    
    # Stratified only
    print("  [2/4] Stratified background only...")
    X_legit = X_train[y_train == 0]
    bg_legit = X_legit[np.random.choice(len(X_legit), min(100, len(X_legit)), replace=False)]
    try:
        exp_strat = shap.TreeExplainer(model, bg_legit)
        shap_strat = exp_strat.shap_values(X_sample)
        if isinstance(shap_strat, list):
            shap_strat = shap_strat[1]
    except:
        shap_strat = shap_std
    strat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_strat).mean(axis=0)
    }).sort_values('importance', ascending=False)
    results.append({
        'method': 'Stratified Background',
        'top_feature': strat_imp.iloc[0]['feature'],
        'mean_importance': strat_imp['importance'].mean()
    })
    
    # Calibrated weights only
    print("  [3/4] Calibrated weights only...")
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    w_legit = n_fraud / (n_legit + n_fraud)
    calib_shap = w_legit * shap_std + (1-w_legit) * shap_std
    calib_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(calib_shap).mean(axis=0)
    }).sort_values('importance', ascending=False)
    results.append({
        'method': 'Calibrated Weights',
        'top_feature': calib_imp.iloc[0]['feature'],
        'mean_importance': calib_imp['importance'].mean()
    })
    
    # Full IC-SHAP
    print("  [4/4] Full IC-SHAP...")
    ic_shap = ImbalanceCalibratedSHAP(model, X_train, y_train)
    ic_values, ic_imp = ic_shap.explain(X_sample, feature_names)
    results.append({
        'method': 'Full IC-SHAP',
        'top_feature': ic_imp.iloc[0]['feature'],
        'mean_importance': ic_imp['importance'].mean()
    })
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_visualizations(output_dir, model_results, std_imp, ic_imp, lime_imp, 
                           xai_quality, cf_stats, comprehensibility_comparison):
    print("\n[LOG] Generating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Model performance
    models = model_results['model'].tolist()
    x = np.arange(len(models))
    axes[0, 0].bar(x - 0.2, model_results['auc_roc'], 0.4, label='AUC-ROC', color='#3498db')
    axes[0, 0].bar(x + 0.2, model_results['auc_prc'], 0.4, label='AUC-PRC', color='#e74c3c')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m[:4].upper() for m in models])
    axes[0, 0].legend()
    axes[0, 0].set_title('Model Performance')
    
    # Feature importance
    top_n = 10
    y_pos = np.arange(top_n)
    axes[0, 1].barh(y_pos - 0.25, std_imp['importance'].head(top_n), 0.25, 
                    label='Standard SHAP', color='#e74c3c')
    axes[0, 1].barh(y_pos, ic_imp['importance'].head(top_n), 0.25,
                    label='IC-SHAP', color='#2ecc71')
    if lime_imp is not None:
        axes[0, 1].barh(y_pos + 0.25, lime_imp['importance'].head(top_n), 0.25,
                        label='LIME', color='#3498db')
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(ic_imp['feature'].head(top_n))
    axes[0, 1].invert_yaxis()
    axes[0, 1].legend()
    axes[0, 1].set_title('Feature Importance Comparison')
    
    # XAI Quality
    metrics = ['Fidelity', 'Stability']
    values = [xai_quality['fidelity'], xai_quality['stability']]
    colors = ['#2ecc71' if v >= t else '#e74c3c' 
              for v, t in zip(values, [Config.TARGET_FIDELITY, Config.TARGET_STABILITY])]
    axes[0, 2].bar(metrics, values, color=colors)
    axes[0, 2].axhline(Config.TARGET_FIDELITY, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('XAI Quality')
    axes[0, 2].set_ylim([0, 1.1])
    
    # Counterfactual stats
    if cf_stats:
        axes[1, 0].bar(['Success', 'Failed'], 
                       [cf_stats['n_successful'], cf_stats['n_total'] - cf_stats['n_successful']],
                       color=['#2ecc71', '#e74c3c'])
        axes[1, 0].set_title(f"Counterfactual Success ({cf_stats['success_rate']:.1f}%)")
    
    # Comprehensibility comparison
    if comprehensibility_comparison:
        methods = ['Standard SHAP', 'IC-SHAP']
        values = [comprehensibility_comparison['standard_shap'], 
                  comprehensibility_comparison['ic_shap']]
        axes[1, 1].bar(methods, values, color=['#e74c3c', '#2ecc71'])
        axes[1, 1].axhline(Config.TARGET_COMPREHENSIBILITY, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Comprehensibility (features for 95% var)')
    
    # IC-SHAP weights
    axes[1, 2].pie([ic_shap.w_legit, ic_shap.w_fraud] if 'ic_shap' in dir() else [0.004, 0.996],
                   labels=['w_legit', 'w_fraud'], autopct='%1.2f%%',
                   colors=['#3498db', '#e74c3c'])
    axes[1, 2].set_title('IC-SHAP Calibration Weights')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[LOG] Saved: {output_dir}/all_results.png")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_complete_experiment(data_path, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("STARTING COMPLETE EXPERIMENT")
    print("="*80)
    
    # Load
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    df = preprocessor.engineer_features(df)
    
    # Split BEFORE encoding
    print("\n[IMPORTANT] Splitting data BEFORE target encoding to prevent leakage")
    train_df, test_df = train_test_split(df, test_size=Config.TEST_SIZE, 
                                          stratify=df['is_fraud'], random_state=RANDOM_STATE)
    train_df, val_df = train_test_split(train_df, test_size=Config.VAL_SIZE,
                                         stratify=train_df['is_fraud'], random_state=RANDOM_STATE)
    
    print(f"  Train: {len(train_df):,} (fraud: {train_df['is_fraud'].sum():,})")
    print(f"  Val: {len(val_df):,} (fraud: {val_df['is_fraud'].sum():,})")
    print(f"  Test: {len(test_df):,} (fraud: {test_df['is_fraud'].sum():,})")
    
    # Fit on train only
    X_train, y_train, feature_names = preprocessor.fit_transform(train_df, fit_encodings=True)
    X_val, y_val, _ = preprocessor.fit_transform(val_df, fit_encodings=False)
    X_test, y_test, _ = preprocessor.fit_transform(test_df, fit_encodings=False)
    
    # Train models
    trainer = ModelTrainer()
    models = trainer.train_all(X_train, y_train, X_val, y_val)
    
    # Evaluate
    model_results = trainer.evaluate(X_test, y_test, feature_names)
    model_results.to_csv(f'{output_dir}/model_performance.csv', index=False)
    print("\n[RESULT] Model Performance:")
    print(model_results[['model', 'auc_roc', 'auc_prc', 'f1_score', 'recall']].to_string(index=False))
    
    # Cross-validation
    print("\n[LOG] Running Out-of-Time CV evaluation...")
    cv_results = run_cross_validation(df, DataPreprocessor(), n_splits=Config.N_FOLDS)
    cv_results.to_csv(f'{output_dir}/cross_validation.csv', index=False)
    print(f"\n[RESULT] CV AUC-ROC: {cv_results['auc_roc'].mean():.4f} (±{cv_results['auc_roc'].std():.4f})")
    
    # Needs entire transformed arrays for downstream XAI evaluation steps
    X_full, y_full, _ = preprocessor.fit_transform(df, fit_encodings=True)
    
    # Standard SHAP
    print("\n" + "="*80)
    print("PHASE 7: SHAP ANALYSIS")
    print("="*80)
    xgb_model = models['xgboost']
    X_sample = X_test[:Config.N_EXPLANATION_SAMPLES]
    
    std_explainer = shap.TreeExplainer(xgb_model)
    std_shap = std_explainer.shap_values(X_sample)
    if isinstance(std_shap, list):
        std_shap = std_shap[1]
    std_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(std_shap).mean(axis=0)
    }).sort_values('importance', ascending=False)
    std_imp.to_csv(f'{output_dir}/standard_shap_importance.csv', index=False)
    print(f"[RESULT] Standard SHAP top feature: {std_imp.iloc[0]['feature']}")
    
    # IC-SHAP
    ic_shap = ImbalanceCalibratedSHAP(xgb_model, X_train, y_train)
    ic_values, ic_imp = ic_shap.explain(X_sample, feature_names)
    ic_imp.to_csv(f'{output_dir}/ic_shap_importance.csv', index=False)
    print(f"[RESULT] IC-SHAP top feature: {ic_imp.iloc[0]['feature']}")
    
    # LIME
    lime_imp = None
    if LIME_AVAILABLE:
        lime_values, lime_imp = run_lime_analysis(xgb_model, X_train, X_sample[:50], feature_names)
        if lime_imp is not None:
            lime_imp.to_csv(f'{output_dir}/lime_importance.csv', index=False)
    
    # Ablation
    ablation_results = run_ablation_study(xgb_model, X_train, y_train, X_test, feature_names)
    ablation_results.to_csv(f'{output_dir}/ablation_study.csv', index=False)
    
    # XAI Quality
    auditor = ExplanationQualityAuditor(xgb_model, feature_names)
    std_quality = auditor.audit(std_explainer, X_sample, "Standard SHAP")
    std_quality['ic_shap_fidelity'] = auditor.evaluate_fidelity(ic_values, 
                                                                 std_explainer.expected_value if not isinstance(std_explainer.expected_value, np.ndarray) else std_explainer.expected_value[1],
                                                                 X_sample)
    std_quality['ic_shap_stability'] = auditor.audit(std_explainer, X_sample, "IC-SHAP")['stability']
    pd.DataFrame([std_quality]).to_csv(f'{output_dir}/xai_quality.csv', index=False)
    
    # Comprehensibility comparison
    std_comp = auditor.evaluate_comprehensibility(std_shap)
    ic_comp = auditor.evaluate_comprehensibility(ic_values)
    comprehensibility_comparison = {
        'standard_shap': std_comp,
        'ic_shap': ic_comp,
        'improvement': (std_comp - ic_comp) / std_comp * 100
    }
    print(f"\n[RESULT] Comprehensibility: Standard={std_comp}, IC-SHAP={ic_comp}")
    
    # Counterfactuals
    cf_gen = CounterfactualGenerator(xgb_model, feature_names)
    cf_results, cf_stats = cf_gen.batch_generate(X_test, y_test)
    cf_results.to_csv(f'{output_dir}/counterfactuals.csv', index=False)
    
    # Visualizations
    generate_visualizations(output_dir, model_results, std_imp, ic_imp, lime_imp,
                          std_quality, cf_stats, comprehensibility_comparison)
    
    # Summary
    summary = {
        'dataset': {
            'name': 'Synthetic Credit Card Transaction Data (Kaggle)',
            'n_samples': int(X_full.shape[0]),
            'n_features': int(X_full.shape[1]),
            'fraud_rate': float(y_full.mean()),
            'imbalance_ratio': float((1-y_full.mean())/y_full.mean()),
            'target_encoding': 'Computed on training folds only (no leakage)'
        },
        'model_performance': {
            'best_model': model_results.loc[model_results['auc_roc'].idxmax(), 'model'],
            'best_auc_roc': float(model_results['auc_roc'].max()),
            'cv_auc_roc_mean': float(cv_results['auc_roc'].mean()),
            'cv_auc_roc_std': float(cv_results['auc_roc'].std())
        },
        'xai_quality': std_quality,
        'counterfactuals': cf_stats,
        'comprehensibility_comparison': comprehensibility_comparison,
        'feature_importance': {
            'standard_shap_top5': std_imp['feature'].head(5).tolist(),
            'ic_shap_top5': ic_imp['feature'].head(5).tolist(),
            'lime_top5': lime_imp['feature'].head(5).tolist() if lime_imp is not None else []
        }
    }
    
    with open(f'{output_dir}/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='fraudTest.csv')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()
    
    summary = run_complete_experiment(args.data, args.output)
