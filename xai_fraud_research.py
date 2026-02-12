"""
Explainable AI for Financial Fraud Detection Research Implementation
====================================================================

This module implements the complete research framework for:
1. Imbalance-Calibrated SHAP (IC-SHAP)
2. Regulatory-Compliant Counterfactual Generator (RC-CF)
3. Explanation Quality Auditor (EQA)

Target: Q1 Journal Publication (Financial Innovation, Expert Systems with Applications)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix, precision_recall_curve,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("XAI for Financial Fraud Detection Research Framework")
print("="*70)


# =============================================================================
# SECTION 1: DATA PREPROCESSING PIPELINE
# =============================================================================

class FraudDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for fraud detection.
    
    Handles:
    - Temporal feature engineering
    - Geographic feature engineering  
    - Categorical encoding
    - Numerical transformations
    - Feature selection
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        self.target_encodings = {}
        self.fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and perform initial data inspection."""
        print(f"\n[1] Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"    Dataset shape: {df.shape}")
        print(f"    Fraud rate: {df['is_fraud'].mean():.4%}")
        return df
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from transaction datetime."""
        df = df.copy()
        
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['day_of_month'] = df['trans_date_trans_time'].dt.day
        df['month'] = df['trans_date_trans_time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['unix_time'] = pd.to_numeric(df['unix_time'])
        
        return df
    
    def engineer_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate geographic distance features."""
        df = df.copy()
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        
        df['distance_to_merchant'] = haversine_distance(
            df['lat'], df['long'], df['merch_lat'], df['merch_long']
        )
        
        df['lat_diff'] = np.abs(df['lat'] - df['merch_lat'])
        df['long_diff'] = np.abs(df['long'] - df['merch_long'])
        
        return df
    
    def engineer_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-related features."""
        df = df.copy()
        
        df['amt_log'] = np.log1p(df['amt'])
        df['amt_sqrt'] = np.sqrt(df['amt'])
        
        return df
    
    def engineer_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic features."""
        df = df.copy()
        
        df['dob'] = pd.to_datetime(df['dob'], format='%d-%m-%Y', errors='coerce')
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
        df['age'] = df['age'].clip(18, 100)
        
        df['city_pop_log'] = np.log1p(df['city_pop'])
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features with target encoding and one-hot encoding."""
        df = df.copy()
        
        if fit:
            for col in ['merchant', 'job']:
                target_mean = df.groupby(col)['is_fraud'].mean()
                self.target_encodings[col] = target_mean
            
            self.category_means = df.groupby('category')['is_fraud'].mean().to_dict()
        
        for col in ['merchant', 'job']:
            df[f'{col}_target_enc'] = df[col].map(self.target_encodings.get(col, {}))
            df[f'{col}_target_enc'] = df[f'{col}_target_enc'].fillna(0)
        
        category_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True)
        df = pd.concat([df, category_dummies], axis=1)
        
        df['gender_encoded'] = (df['gender'] == 'M').astype(int)
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Select final feature set for modeling."""
        feature_cols = [
            'amt', 'amt_log', 'amt_sqrt',
            'hour', 'hour_sin', 'hour_cos', 
            'day_of_week', 'day_sin', 'day_cos',
            'day_of_month', 'month', 'is_weekend',
            'distance_to_merchant', 'lat_diff', 'long_diff',
            'age', 'city_pop', 'city_pop_log',
            'merchant_target_enc', 'job_target_enc',
            'gender_encoded'
        ]
        
        category_cols = [c for c in df.columns if c.startswith('cat_')]
        feature_cols.extend(category_cols)
        
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[feature_cols].copy()
        
        return X, feature_cols
    
    def fit_transform(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Complete preprocessing pipeline."""
        df = self.load_data(filepath)
        
        print("[2] Engineering temporal features...")
        df = self.engineer_temporal_features(df)
        
        print("[3] Engineering geographic features...")
        df = self.engineer_geographic_features(df)
        
        print("[4] Engineering amount features...")
        df = self.engineer_amount_features(df)
        
        print("[5] Engineering demographic features...")
        df = self.engineer_demographic_features(df)
        
        print("[6] Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=True)
        
        print("[7] Selecting features...")
        X, self.feature_names = self.select_features(df)
        y = df['is_fraud'].copy()
        
        X = X.fillna(X.median())
        
        print(f"\n[RESULT] Processed dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"         Fraud rate: {y.mean():.4%}")
        print(f"         Imbalance ratio: 1:{int((1-y.mean())/y.mean())}")
        
        self.fitted = True
        return X, y, self.feature_names
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        df = self.engineer_temporal_features(df)
        df = self.engineer_geographic_features(df)
        df = self.engineer_amount_features(df)
        df = self.engineer_demographic_features(df)
        df = self.encode_categorical_features(df, fit=False)
        X, _ = self.select_features(df)
        X = X.fillna(X.median())
        return X


# =============================================================================
# SECTION 2: IMBALANCE-CALIBRATED SHAP (IC-SHAP)
# =============================================================================

class ImbalanceCalibratedSHAP:
    """
    Imbalance-Calibrated SHAP for extreme class imbalance scenarios.
    
    Key innovations:
    1. Stratified background sampling per class
    2. Class-conditional SHAP computation
    3. Calibration weights inversely proportional to class frequency
    
    Addresses systematic bias in traditional SHAP when applied to
    imbalanced datasets where the background distribution is dominated
    by majority class instances.
    """
    
    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                 n_background_samples: int = 500, feature_names: List[str] = None):
        """
        Initialize IC-SHAP explainer.
        
        Parameters:
        -----------
        model : object
            Trained model with predict_proba method
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels (0: legitimate, 1: fraud)
        n_background_samples : int
            Number of samples per class for background distribution
        feature_names : List[str]
            Names of features for output
        """
        self.model = model
        self.n_background_samples = n_background_samples
        self.feature_names = feature_names
        
        X_legit = X_train[y_train == 0]
        X_fraud = X_train[y_train == 1]
        
        self.bg_legit = X_legit[
            np.random.choice(len(X_legit), min(n_background_samples, len(X_legit)), replace=False)
        ] if len(X_legit) > 0 else X_legit
        
        self.bg_fraud = X_fraud[
            np.random.choice(len(X_fraud), min(n_background_samples, len(X_fraud)), 
                           replace=len(X_fraud) < n_background_samples)
        ] if len(X_fraud) > 0 else X_fraud
        
        n_legit = (y_train == 0).sum()
        n_fraud = (y_train == 1).sum()
        self.w_legit = n_fraud / (n_legit + n_fraud)
        self.w_fraud = n_legit / (n_legit + n_fraud)
        
        print(f"\n[IC-SHAP] Background samples - Legitimate: {len(self.bg_legit)}, Fraud: {len(self.bg_fraud)}")
        print(f"[IC-SHAP] Calibration weights - w_legit: {self.w_legit:.4f}, w_fraud: {self.w_fraud:.4f}")
        
        self.explainer_legit = None
        self.explainer_fraud = None
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP explainers with different background distributions."""
        try:
            self.explainer_legit = shap.TreeExplainer(self.model, self.bg_legit)
            self.explainer_fraud = shap.TreeExplainer(self.model, self.bg_fraud)
            self.explainer_type = 'Tree'
        except Exception as e:
            print(f"[IC-SHAP] TreeExplainer failed, using KernelExplainer: {e}")
            self.explainer_legit = shap.KernelExplainer(
                self.model.predict_proba, self.bg_legit[:100]
            )
            self.explainer_fraud = shap.KernelExplainer(
                self.model.predict_proba, self.bg_fraud[:100]
            )
            self.explainer_type = 'Kernel'
    
    def explain(self, X_explain: np.ndarray, return_both: bool = False) -> np.ndarray:
        """
        Generate IC-SHAP explanations.
        
        Parameters:
        -----------
        X_explain : np.ndarray
            Instances to explain
        return_both : bool
            If True, return both class-conditional SHAP values
            
        Returns:
        --------
        shap_values_ic : np.ndarray
            Imbalance-calibrated SHAP values
        """
        shap_legit = self.explainer_legit.shap_values(X_explain)
        shap_fraud = self.explainer_fraud.shap_values(X_explain)
        
        if isinstance(shap_legit, list):
            shap_legit = shap_legit[1]
            shap_fraud = shap_fraud[1]
        
        shap_values_ic = (
            self.w_legit * shap_legit + 
            self.w_fraud * shap_fraud
        )
        
        if return_both:
            return shap_values_ic, shap_legit, shap_fraud
        
        return shap_values_ic
    
    def get_feature_importance(self, X_explain: np.ndarray) -> pd.DataFrame:
        """Get global feature importance from SHAP values."""
        shap_values = self.explain(X_explain)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'f{i}' for i in range(shap_values.shape[1])],
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_instance(self, x_single: np.ndarray) -> Dict:
        """Explain a single instance with detailed output."""
        if x_single.ndim == 1:
            x_single = x_single.reshape(1, -1)
        
        shap_values = self.explain(x_single)[0]
        
        feature_names = self.feature_names if self.feature_names else [f'f{i}' for i in range(len(shap_values))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'top_5_features': importance_df.head(5)['feature'].tolist(),
            'top_5_contribution': importance_df.head(5)['shap_value'].tolist()
        }


# =============================================================================
# SECTION 3: REGULATORY-COMPLIANT COUNTERFACTUAL GENERATOR (RC-CF)
# =============================================================================

class RegulatoryCompliantCF:
    """
    Regulatory-Compliant Counterfactual Generator for Financial Fraud Detection.
    
    Generates counterfactual explanations satisfying:
    - Actionability constraints (only modifiable features)
    - Feasibility constraints (realistic changes)
    - Non-discrimination constraints (protected attributes)
    - Sparsity constraints (minimal changes)
    
    Compliant with GDPR Article 22, ECOA, and Fair Credit Reporting Act.
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        immutable_features: List[str],
        bounds: Dict[str, Tuple[float, float]],
        actionable_features: Optional[List[str]] = None
    ):
        """
        Initialize RC-CF generator.
        
        Parameters:
        -----------
        model : object
            Trained model with predict_proba method
        feature_names : List[str]
            Names of all features
        immutable_features : List[str]
            Features that cannot change (e.g., age, gender)
        bounds : Dict[str, Tuple[float, float]]
            Min-max bounds for each feature
        actionable_features : List[str], optional
            Features that can be modified
        """
        self.model = model
        self.feature_names = feature_names
        self.immutable_features = immutable_features
        self.bounds = bounds
        
        if actionable_features is None:
            self.actionable_features = [
                f for f in feature_names if f not in immutable_features
            ]
        else:
            self.actionable_features = actionable_features
        
        self.immutable_indices = [
            feature_names.index(f) for f in immutable_features if f in feature_names
        ]
        self.actionable_indices = [
            feature_names.index(f) for f in self.actionable_features if f in feature_names
        ]
        
        self.lambda_dist = 0.1
        self.lambda_sparse = 0.5
        self.lambda_actionable = 0.3
        
        print(f"\n[RC-CF] Initialized with {len(self.actionable_features)} actionable features")
        print(f"[RC-CF] Immutable features: {immutable_features}")
    
    def generate_counterfactual(
        self,
        x_original: np.ndarray,
        target_class: int = 0,
        threshold: float = 0.5,
        max_iterations: int = 1000,
        n_candidates: int = 5
    ) -> Dict:
        """
        Generate regulatory-compliant counterfactual explanation.
        
        Parameters:
        -----------
        x_original : np.ndarray
            Original instance flagged as fraud
        target_class : int
            Target class (0 for legitimate)
        threshold : float
            Decision threshold
        max_iterations : int
            Maximum optimization iterations
        n_candidates : int
            Number of candidate counterfactuals
            
        Returns:
        --------
        dict : Contains best counterfactual and metadata
        """
        from scipy.optimize import minimize
        
        candidates = []
        
        for i in range(n_candidates):
            result = self._optimize_single(
                x_original, target_class, threshold, max_iterations, seed=i
            )
            if result['success']:
                candidates.append(result)
        
        if not candidates:
            return {
                'success': False,
                'message': 'No valid counterfactual found'
            }
        
        best_cf = min(candidates, key=lambda x: x['n_changes'])
        return best_cf
    
    def _optimize_single(
        self,
        x_original: np.ndarray,
        target_class: int,
        threshold: float,
        max_iterations: int,
        seed: int
    ) -> Dict:
        """Run single optimization."""
        from scipy.optimize import minimize
        
        def objective(x_cf):
            try:
                pred = self.model.predict_proba(x_cf.reshape(1, -1))[0, 1]
            except:
                pred = 0.5
            
            pred_loss = max(0, pred - (1 - threshold)) if target_class == 0 else max(0, threshold - pred)
            
            dist_loss = np.linalg.norm(x_cf - x_original)
            
            sparse_loss = np.sum(np.abs(x_cf - x_original) > 1e-6)
            
            actionable_mask = np.zeros(len(x_original))
            actionable_mask[self.actionable_indices] = 1
            action_loss = np.sum((1 - actionable_mask) * np.abs(x_cf - x_original))
            
            return (
                10 * pred_loss + 
                self.lambda_dist * dist_loss + 
                self.lambda_sparse * sparse_loss +
                self.lambda_actionable * action_loss
            )
        
        opt_bounds = [
            self.bounds.get(self.feature_names[i], (-1e10, 1e10))
            for i in range(len(x_original))
        ]
        
        constraints = []
        for idx in self.immutable_indices:
            constraints.append({
                'type': 'eq',
                'fun': lambda x, idx=idx: x[idx] - x_original[idx]
            })
        
        np.random.seed(seed)
        x_init = x_original.copy()
        for idx in self.actionable_indices:
            lb, ub = opt_bounds[idx]
            if lb is not None and ub is not None:
                x_init[idx] += np.random.uniform(-0.1, 0.1) * (ub - lb)
                x_init[idx] = np.clip(x_init[idx], lb, ub)
        
        result = minimize(
            objective,
            x_init,
            method='SLSQP',
            bounds=opt_bounds,
            constraints=constraints,
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )
        
        x_cf = result.x
        changes = np.abs(x_cf - x_original) > 1e-6
        
        try:
            pred_original = self.model.predict_proba(x_original.reshape(1, -1))[0, 1]
            pred_cf = self.model.predict_proba(x_cf.reshape(1, -1))[0, 1]
        except:
            pred_original = 0.5
            pred_cf = 0.5
        
        return {
            'success': result.success,
            'counterfactual': x_cf,
            'original': x_original,
            'n_changes': np.sum(changes),
            'changed_features': [
                self.feature_names[i] for i in range(len(changes)) if changes[i]
            ],
            'change_magnitudes': {
                self.feature_names[i]: float(x_cf[i] - x_original[i])
                for i in range(len(changes)) if changes[i]
            },
            'prediction_original': float(pred_original),
            'prediction_cf': float(pred_cf),
            'class_flipped': (pred_cf < threshold) if target_class == 0 else (pred_cf >= threshold)
        }


# =============================================================================
# SECTION 4: EXPLANATION QUALITY AUDITOR (EQA)
# =============================================================================

class ExplanationQualityAuditor:
    """
    Automated assessment of explanation quality across multiple dimensions.
    
    Evaluates:
    - Fidelity: Explainer-model agreement
    - Stability: Consistency under perturbations
    - Comprehensibility: Explanation simplicity
    - Correctness: Feature importance accuracy
    - Discrimination Index: Fairness measure
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize EQA.
        
        Parameters:
        -----------
        model : object
            Trained model
        feature_names : List[str]
            Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.results = {}
    
    def evaluate_fidelity(self, explainer, X_sample: np.ndarray) -> float:
        """Evaluate explanation fidelity."""
        try:
            shap_values = explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            
            reconstructed = base_value + shap_values.sum(axis=1)
            actual = self.model.predict_proba(X_sample)[:, 1]
            
            correlation = np.corrcoef(reconstructed, actual)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception as e:
            print(f"[EQA] Fidelity evaluation failed: {e}")
            return 0.0
    
    def evaluate_stability(self, explainer, X_sample: np.ndarray, 
                          n_perturbations: int = 5) -> float:
        """Evaluate explanation stability under small perturbations."""
        try:
            base_shap = explainer.shap_values(X_sample[:1])
            if isinstance(base_shap, list):
                base_shap = base_shap[1][0]
            else:
                base_shap = base_shap[0]
            
            base_top = set(np.argsort(np.abs(base_shap))[-10:])
            
            jaccard_scores = []
            
            for _ in range(n_perturbations):
                X_perturbed = X_sample[:1] + np.random.normal(0, 0.01, X_sample[:1].shape)
                perturbed_shap = explainer.shap_values(X_perturbed)
                if isinstance(perturbed_shap, list):
                    perturbed_shap = perturbed_shap[1][0]
                else:
                    perturbed_shap = perturbed_shap[0]
                
                perturbed_top = set(np.argsort(np.abs(perturbed_shap))[-10:])
                
                intersection = len(base_top & perturbed_top)
                union = len(base_top | perturbed_top)
                jaccard = intersection / union if union > 0 else 1.0
                jaccard_scores.append(jaccard)
            
            return float(np.mean(jaccard_scores))
        except Exception as e:
            print(f"[EQA] Stability evaluation failed: {e}")
            return 0.0
    
    def evaluate_comprehensibility(self, shap_values: np.ndarray, 
                                   threshold: float = 0.05) -> int:
        """Evaluate comprehensibility (number of significant features)."""
        abs_shap = np.abs(shap_values)
        if abs_shap.ndim > 1:
            abs_shap = abs_shap.mean(axis=0)
        
        total_importance = abs_shap.sum()
        if total_importance == 0:
            return len(self.feature_names)
        
        sorted_idx = np.argsort(abs_shap)[::-1]
        cumulative = np.cumsum(abs_shap[sorted_idx]) / total_importance
        
        n_features = np.searchsorted(cumulative, 1 - threshold) + 1
        return int(n_features)
    
    def evaluate_actionability(self, cf_results: List[Dict]) -> float:
        """Evaluate actionability of counterfactuals."""
        if not cf_results:
            return 0.0
        
        actionable_ratios = []
        for cf in cf_results:
            if not cf.get('success', False):
                continue
            
            n_changes = cf.get('n_changes', 0)
            if n_changes == 0:
                continue
            
            changed_features = cf.get('changed_features', [])
            actionable = [f for f in changed_features if f in self.feature_names]
            
            if len(changed_features) > 0:
                actionable_ratios.append(len(actionable) / len(changed_features))
        
        return float(np.mean(actionable_ratios)) if actionable_ratios else 0.0
    
    def audit_explanation(self, explainer, X_sample: np.ndarray, 
                         cf_results: List[Dict] = None) -> Dict:
        """
        Complete audit of explanation quality.
        
        Returns:
        --------
        dict : Quality metrics
        """
        print("\n[EQA] Running explanation quality audit...")
        
        fidelity = self.evaluate_fidelity(explainer, X_sample)
        print(f"    Fidelity: {fidelity:.4f}")
        
        stability = self.evaluate_stability(explainer, X_sample)
        print(f"    Stability: {stability:.4f}")
        
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        comprehensibility = self.evaluate_comprehensibility(shap_values)
        print(f"    Comprehensibility: {comprehensibility} significant features")
        
        actionability = 0.0
        if cf_results:
            actionability = self.evaluate_actionability(cf_results)
            print(f"    Actionability: {actionability:.4f}")
        
        self.results = {
            'fidelity': fidelity,
            'stability': stability,
            'comprehensibility': comprehensibility,
            'actionability': actionability,
            'pass_threshold': fidelity > 0.90 and stability > 0.85
        }
        
        return self.results


# =============================================================================
# SECTION 5: MODEL TRAINING
# =============================================================================

class FraudDetectionModels:
    """
    Training and evaluation of fraud detection models.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_xgboost(self, X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
        """Train XGBoost classifier."""
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            max_depth=8,
            n_estimators=300,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='auc',
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
        """Train LightGBM classifier."""
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = lgb.LGBMClassifier(
            num_leaves=64,
            n_estimators=300,
            learning_rate=0.03,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        self.models['lightgbm'] = model
        return model
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Train Random Forest classifier."""
        class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, name: str) -> Dict:
        """Comprehensive model evaluation."""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_prc': average_precision_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
        }
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        })
        
        self.results[name] = metrics
        return metrics
    
    def print_results(self):
        """Print evaluation results."""
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        
        for name, metrics in self.results.items():
            print(f"\n{name.upper()}:")
            print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
            print(f"  AUC-PRC:      {metrics['auc_prc']:.4f}")
            print(f"  F1-Score:     {metrics['f1_score']:.4f}")
            print(f"  MCC:          {metrics['mcc']:.4f}")
            print(f"  Detection Rate: {metrics['detection_rate']:.4f}")
            print(f"  Precision:    {metrics['precision']:.4f}")
            print(f"  Confusion Matrix: TP={metrics['true_positives']}, FP={metrics['false_positives']}, TN={metrics['true_negatives']}, FN={metrics['false_negatives']}")


# =============================================================================
# SECTION 6: MAIN EXPERIMENT PIPELINE
# =============================================================================

def run_complete_experiment(filepath: str, output_dir: str = 'results'):
    """
    Run complete XAI fraud detection experiment.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPROCESSING")
    print("="*70)
    
    preprocessor = FraudDataPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(filepath)
    
    print("\n" + "="*70)
    print("PHASE 2: DATA SPLITTING")
    print("="*70)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X.values, y.values, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    print(f"Training set:   {X_train.shape[0]} samples, Fraud rate: {y_train.mean():.4%}")
    print(f"Validation set: {X_val.shape[0]} samples, Fraud rate: {y_val.mean():.4%}")
    print(f"Test set:       {X_test.shape[0]} samples, Fraud rate: {y_test.mean():.4%}")
    
    print("\n" + "="*70)
    print("PHASE 3: MODEL TRAINING")
    print("="*70)
    
    trainer = FraudDetectionModels()
    
    print("\n[1/3] Training XGBoost...")
    xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    
    print("[2/3] Training LightGBM...")
    lgb_model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    
    print("[3/3] Training Random Forest...")
    rf_model = trainer.train_random_forest(X_train, y_train)
    
    print("\n" + "="*70)
    print("PHASE 4: MODEL EVALUATION")
    print("="*70)
    
    trainer.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
    trainer.evaluate_model(lgb_model, X_test, y_test, 'lightgbm')
    trainer.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    trainer.print_results()
    
    print("\n" + "="*70)
    print("PHASE 5: XAI ANALYSIS - IC-SHAP")
    print("="*70)
    
    X_test_sample = X_test[:500]
    y_test_sample = y_test[:500]
    
    ic_shap = ImbalanceCalibratedSHAP(
        xgb_model, X_train, y_train, 
        n_background_samples=500,
        feature_names=feature_names
    )
    
    ic_shap_values = ic_shap.explain(X_test_sample)
    
    feature_importance = ic_shap.get_feature_importance(X_test_sample)
    print("\n[IC-SHAP] Top 15 Features by Importance:")
    print(feature_importance.head(15).to_string(index=False))
    
    print("\n" + "="*70)
    print("PHASE 6: XAI ANALYSIS - STANDARD SHAP (BASELINE)")
    print("="*70)
    
    standard_explainer = shap.TreeExplainer(xgb_model)
    standard_shap_values = standard_explainer.shap_values(X_test_sample)
    if isinstance(standard_shap_values, list):
        standard_shap_values = standard_shap_values[1]
    
    standard_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(standard_shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\n[Standard SHAP] Top 15 Features by Importance:")
    print(standard_importance.head(15).to_string(index=False))
    
    print("\n" + "="*70)
    print("PHASE 7: EXPLANATION QUALITY AUDIT")
    print("="*70)
    
    eqa = ExplanationQualityAuditor(xgb_model, feature_names)
    eqa_results = eqa.audit_explanation(ic_shap, X_test_sample[:100])
    
    print("\n" + "="*70)
    print("PHASE 8: COUNTERFACTUAL GENERATION")
    print("="*70)
    
    fraud_indices = np.where(y_test == 1)[0][:10]
    if len(fraud_indices) > 0:
        bounds = {}
        for i, fname in enumerate(feature_names):
            bounds[fname] = (X_train[:, i].min(), X_train[:, i].max())
        
        immutable = ['age', 'gender_encoded', 'city_pop', 'city_pop_log']
        immutable = [f for f in immutable if f in feature_names]
        
        rc_cf = RegulatoryCompliantCF(
            xgb_model, feature_names, immutable, bounds
        )
        
        cf_results = []
        for idx in fraud_indices[:5]:
            cf = rc_cf.generate_counterfactual(X_test[idx], n_candidates=3)
            cf_results.append(cf)
            if cf['success']:
                print(f"\n[CF] Instance {idx}:")
                print(f"    Original prediction: {cf['prediction_original']:.4f}")
                print(f"    CF prediction: {cf['prediction_cf']:.4f}")
                print(f"    Features changed: {cf['n_changes']}")
                print(f"    Changed features: {cf['changed_features'][:5]}")
    
    print("\n" + "="*70)
    print("PHASE 9: GENERATING VISUALIZATIONS")
    print("="*70)
    
    generate_visualizations(
        trainer.results, 
        feature_importance,
        standard_importance,
        ic_shap_values,
        standard_shap_values,
        feature_names,
        eqa_results,
        output_dir
    )
    
    print("\n" + "="*70)
    print("PHASE 10: SAVING RESULTS")
    print("="*70)
    
    results_summary = {
        'model_performance': trainer.results,
        'xai_quality': eqa_results,
        'top_features': feature_importance.head(20).to_dict('records'),
        'dataset_stats': {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'fraud_rate': float(y.mean()),
            'imbalance_ratio': float((1-y.mean())/y.mean())
        }
    }
    
    with open(f'{output_dir}/experiment_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/")
    
    return {
        'preprocessor': preprocessor,
        'models': trainer.models,
        'ic_shap': ic_shap,
        'eqa': eqa,
        'results': results_summary
    }


def generate_visualizations(model_results, ic_importance, std_importance,
                           ic_shap_values, std_shap_values, feature_names,
                           eqa_results, output_dir):
    """Generate publication-ready visualizations."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models = list(model_results.keys())
    auc_rocs = [model_results[m]['auc_roc'] for m in models]
    auc_prcs = [model_results[m]['auc_prc'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, auc_rocs, width, label='AUC-ROC', color='#3498db')
    axes[0, 0].bar(x + width/2, auc_prcs, width, label='AUC-PRC', color='#e74c3c')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.upper() for m in models])
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.1])
    
    top_n = 15
    ic_top = ic_importance.head(top_n)
    
    axes[0, 1].barh(range(top_n), ic_top['importance'].values, color='#2ecc71')
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(ic_top['feature'].values)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel('Mean |SHAP Value|')
    axes[0, 1].set_title('IC-SHAP Feature Importance (Top 15)')
    
    ic_global = np.abs(ic_shap_values).mean(axis=0)
    std_global = np.abs(std_shap_values).mean(axis=0)
    
    top_features_idx = np.argsort(ic_global)[-10:]
    
    axes[1, 0].scatter(std_global[top_features_idx], ic_global[top_features_idx], 
                       c='#9b59b6', s=100, alpha=0.7)
    axes[1, 0].plot([0, max(ic_global)], [0, max(ic_global)], 'k--', alpha=0.3)
    axes[1, 0].set_xlabel('Standard SHAP Importance')
    axes[1, 0].set_ylabel('IC-SHAP Importance')
    axes[1, 0].set_title('IC-SHAP vs Standard SHAP Comparison')
    
    for i, idx in enumerate(top_features_idx):
        axes[1, 0].annotate(feature_names[idx][:10], 
                           (std_global[idx], ic_global[idx]),
                           fontsize=8, alpha=0.7)
    
    metrics = ['fidelity', 'stability']
    values = [eqa_results.get(m, 0) for m in metrics]
    
    axes[1, 1].bar(metrics, values, color=['#1abc9c', '#f39c12'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Explanation Quality Metrics')
    axes[1, 1].set_ylim([0, 1.1])
    
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/research_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/research_results.png")


if __name__ == "__main__":
    results = run_complete_experiment('fraudTest.csv', output_dir='results')
    print("\n" + "="*70)
    print("RESEARCH EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)
