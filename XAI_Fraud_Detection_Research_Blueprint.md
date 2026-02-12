# Research Blueprint: Explainable Artificial Intelligence (XAI) for Financial Fraud Detection

**A Comprehensive Research Framework for Q1 Journal Publication**

**Target Venues:** Financial Innovation (Q1) | Expert Systems with Applications (Q1) | IEEE TNNLS

**Dataset:** Kaggle Credit Card Fraud Detection Dataset  
**Scale:** 556,861 Transactions | 0.38% Fraud Rate | Extreme Class Imbalance

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
   - 1.1 [Research Objectives](#11-research-objectives)
2. [Research Problem Statement](#2-research-problem-statement)
   - 2.1 [Problem Decomposition](#21-problem-decomposition)
   - 2.2 [Research Questions](#22-research-questions)
3. [Literature Review Framework](#3-literature-review-framework)
   - 3.1 [Financial Fraud Detection: State of the Art](#31-financial-fraud-detection-state-of-the-art)
   - 3.2 [Explainable AI: Methods and Applications](#32-explainable-ai-methods-and-applications)
   - 3.3 [Identified Research Gaps](#33-identified-research-gaps)
4. [Proposed Methodology](#4-proposed-methodology)
   - 4.1 [Component 1: Imbalance-Calibrated SHAP (IC-SHAP)](#41-component-1-imbalance-calibrated-shap-ic-shap)
   - 4.2 [Component 2: Regulatory-Compliant Counterfactual Generator (RC-CF)](#42-component-2-regulatory-compliant-counterfactual-generator-rc-cf)
   - 4.3 [Component 3: Explanation Quality Auditor (EQA)](#43-component-3-explanation-quality-auditor-eqa)
5. [Technical Implementation Details](#5-technical-implementation-details)
   - 5.1 [Data Preprocessing Pipeline](#51-data-preprocessing-pipeline)
   - 5.2 [Model Architecture Specifications](#52-model-architecture-specifications)
   - 5.3 [XAI Implementation Specifications](#53-xai-implementation-specifications)
6. [Experimental Design](#6-experimental-design)
   - 6.1 [Dataset Configuration](#61-dataset-configuration)
   - 6.2 [Baseline Models and Methods](#62-baseline-models-and-methods)
   - 6.3 [Evaluation Metrics](#63-evaluation-metrics)
   - 6.4 [Ablation Study Design](#64-ablation-study-design)
7. [Python Implementation Templates](#7-python-implementation-templates)
   - 7.1 [IC-SHAP Implementation Template](#71-ic-shap-implementation-template)
   - 7.2 [RC-CF Implementation Template](#72-rc-cf-implementation-template)
   - 7.3 [Complete Experiment Pipeline Template](#73-complete-experiment-pipeline-template)
8. [Expected Contributions](#8-expected-contributions)
   - 8.1 [Theoretical Contributions](#81-theoretical-contributions)
   - 8.2 [Methodological Contributions](#82-methodological-contributions)
   - 8.3 [Empirical Contributions](#83-empirical-contributions)
9. [Target Paper Structure](#9-target-paper-structure)
10. [Research Timeline](#10-research-timeline)
11. [Required Dependencies and Resources](#11-required-dependencies-and-resources)
12. [Target Journal Specifications](#12-target-journal-specifications)
13. [Instructions for AI Execution](#13-instructions-for-ai-execution)

---

## 1. Executive Summary

This research blueprint provides a comprehensive framework for developing and publishing a Q1-level research paper on Explainable Artificial Intelligence (XAI) for Financial Fraud Detection. The document is designed to be handed to an AI model for complete execution, covering all aspects from problem formulation to paper submission. The research addresses a critical gap in the intersection of financial fraud detection and explainable AI, specifically targeting the challenge of extreme class imbalance (0.38% fraud rate) while maintaining regulatory-compliant interpretability.

The proposed research introduces three novel contributions: (1) an explanation framework specifically designed for extreme imbalanced data that overcomes the bias issues in traditional XAI methods; (2) a hybrid SHAP-Counterfactual explanation framework that provides both local and global interpretability suitable for financial regulatory requirements; and (3) a decision audit trail system that enables financial institutions to provide transparent explanations for fraud detection decisions. These contributions address the fundamental tension between model complexity, detection accuracy, and regulatory interpretability requirements in financial services.

### 1.1 Research Objectives

1. Develop an XAI framework robust to extreme class imbalance (0.38% fraud rate)
2. Create counterfactual explanations meeting financial regulatory standards (GDPR, ECOA, Fair Credit Reporting Act)
3. Achieve Pareto-optimal balance between model interpretability and detection accuracy
4. Provide actionable insights for financial institutions implementing AI-based fraud detection systems
5. Establish benchmarks for XAI methods in financial fraud detection context

---

## 2. Research Problem Statement

Financial fraud detection presents a unique challenge at the intersection of machine learning, regulatory compliance, and practical deployment. While deep learning models have achieved remarkable detection accuracy, their "black-box" nature conflicts with financial regulations requiring explainability. This problem is further compounded by extreme class imbalance, where fraud cases constitute only 0.38% of transactions in typical datasets, creating significant challenges for both model training and explanation generation.

Traditional XAI methods like SHAP and LIME were developed primarily for balanced classification scenarios and exhibit systematic biases when applied to extreme imbalanced datasets. These biases manifest as skewed feature importance rankings, unstable explanations for minority class instances, and counterfactual explanations that may suggest unrealistic or impossible feature modifications. For financial institutions, these limitations translate into regulatory risk, potential discrimination claims, and erosion of customer trust.

### 2.1 Problem Decomposition

| Problem Dimension | Challenge | Research Focus |
|-------------------|-----------|----------------|
| Class Imbalance | 0.38% fraud rate creates biased explanations | Imbalance-aware explanation methods |
| Regulatory Compliance | GDPR Article 22, ECOA requirements | Audit-ready explanation formats |
| Model-Explanation Trade-off | Complex models vs. interpretability | Pareto-optimal framework design |
| Feature Interactions | Non-linear relationships in fraud patterns | Interaction-aware explanations |
| Temporal Dynamics | Evolving fraud tactics over time | Time-sensitive explanation stability |

### 2.2 Research Questions

This research addresses the following primary questions that guide the methodological development and experimental design:

- **RQ1:** How do existing XAI methods (SHAP, LIME, Integrated Gradients) perform under extreme class imbalance conditions, and what systematic biases emerge in their explanations?
- **RQ2:** What modifications to explanation algorithms are necessary to achieve reliable feature importance rankings for minority class (fraud) instances?
- **RQ3:** How can counterfactual explanations be generated that satisfy both regulatory requirements (actionability, feasibility) and maintain fidelity to the underlying model?
- **RQ4:** What is the optimal balance between model complexity, detection performance, and explanation quality for financial fraud detection systems?
- **RQ5:** How do different explanation formats (feature importance, counterfactuals, decision rules) impact user trust and decision-making in fraud investigation contexts?

---

## 3. Literature Review Framework

A comprehensive literature review forms the foundation of this research, establishing the theoretical context and identifying gaps that justify the proposed contributions. The review should systematically analyze existing work across four interconnected domains: financial fraud detection methods, explainable AI techniques, imbalanced learning approaches, and regulatory frameworks for algorithmic decision-making.

### 3.1 Financial Fraud Detection: State of the Art

The fraud detection literature has evolved from rule-based systems through traditional machine learning to deep learning approaches. Key works include:

| Method Category | Key References | Strengths | Limitations |
|-----------------|----------------|-----------|-------------|
| Traditional ML (RF, XGBoost) | Bhattacharyya et al. (2011) | Interpretable, efficient | Limited pattern recognition |
| Deep Learning (MLP, CNN) | Roy et al. (2018) | High accuracy, feature learning | Black-box nature |
| Graph Neural Networks | Liu et al. (2022) | Network structure exploitation | Complexity, interpretability |
| Ensemble Methods | Zhou et al. (2020) | Robust performance | Aggregation complexity |
| Hybrid Approaches | Varmedja et al. (2019) | Combined strengths | Integration challenges |

### 3.2 Explainable AI: Methods and Applications

The XAI landscape has expanded significantly, with methods categorized along multiple dimensions: model-agnostic vs. model-specific, global vs. local explanations, and post-hoc vs. intrinsic interpretability. The following taxonomy provides structure for literature analysis:

| XAI Method | Type | Output Format | Key Citations |
|------------|------|---------------|---------------|
| SHAP (Shapley Values) | Model-agnostic | Feature importance scores | Lundberg & Lee (2017) |
| LIME | Model-agnostic | Local linear approximations | Ribeiro et al. (2016) |
| Integrated Gradients | Model-specific (DNN) | Attribution maps | Sundararajan et al. (2017) |
| Counterfactuals | Model-agnostic | What-if scenarios | Wachter et al. (2018) |
| Attention Mechanisms | Intrinsic | Attention weights | Bahdanau et al. (2015) |
| Decision Trees/Rules | Intrinsic | Rule sets | Lakkaraju et al. (2016) |

### 3.3 Identified Research Gaps

Through systematic analysis of existing literature, the following gaps emerge as opportunities for novel contributions:

1. **Gap 1:** Limited research on XAI methods specifically designed for extreme class imbalance scenarios. Most XAI benchmarks use balanced datasets (e.g., Iris, Wine, Adult Census).

2. **Gap 2:** Lack of unified frameworks combining multiple explanation modalities (feature importance, counterfactuals, rules) for financial decision support.

3. **Gap 3:** Insufficient attention to regulatory compliance requirements in explanation generation (actionability, feasibility, non-discrimination).

4. **Gap 4:** Absence of comprehensive evaluation frameworks for XAI quality in fraud detection contexts beyond standard metrics.

---

## 4. Proposed Methodology

The proposed methodology introduces a novel framework called **Imbalance-Aware Explainable Fraud Detection (IAE-FD)** that addresses the identified gaps through three integrated components: (1) an Imbalance-Calibrated SHAP (IC-SHAP) algorithm, (2) a Regulatory-Compliant Counterfactual Generator (RC-CF), and (3) an Explanation Quality Auditor (EQA) module. These components work synergistically to provide accurate, reliable, and compliant explanations for fraud detection decisions.

### 4.1 Component 1: Imbalance-Calibrated SHAP (IC-SHAP)

Traditional SHAP explanations suffer from systematic bias when applied to imbalanced datasets because the background distribution used for feature attribution is dominated by majority class instances. IC-SHAP addresses this through stratified sampling, class-specific background distributions, and calibration adjustments.

**Mathematical Formulation:**

Standard SHAP:
```
φᵢ(f,x) = Σ[|S|!(n-|S|-1)!/n!] × [fₓ(S∪{i}) - fₓ(S)]
```

Imbalance-Calibrated SHAP:
```
φᵢ^(IC)(f,x) = w_c × φᵢ(f,x|D_c)
```
where D_c is the class-specific background distribution and w_c is a calibration weight inversely proportional to class frequency.

**Algorithm Steps:**

1. **Stratified Background Sampling:** Create separate background distributions D₀ (legitimate) and D₁ (fraud) with equal representation from each class.

2. **Class-Conditional SHAP Computation:** Calculate separate SHAP values using each background distribution: φᵢ⁽⁰⁾ using D₀ and φᵢ⁽¹⁾ using D₁.

3. **Calibration Weight Calculation:** Compute:
   - w₀ = N₁/(N₀+N₁)
   - w₁ = N₀/(N₀+N₁)
   - where N_c is the sample count for class c

4. **Weighted Aggregation:** Combine class-conditional SHAP values:
   ```
   φᵢ^(IC) = w₀ × φᵢ⁽⁰⁾ + w₁ × φᵢ⁽¹⁾
   ```
   with instance-class alignment.

### 4.2 Component 2: Regulatory-Compliant Counterfactual Generator (RC-CF)

Counterfactual explanations answer "what-if" questions by identifying minimal changes to input features that would alter the model's prediction. For financial applications, counterfactuals must satisfy additional constraints related to actionability, feasibility, and regulatory compliance.

**Optimization Formulation:**

```
minimize: L_pred(c,x') + λ₁L_dist(x,x') + λ₂L_sparse(x,x') + λ₃L_actionable(x,x')

subject to:
    (1) f(x') ≥ τ          (prediction flip)
    (2) x'_i ∈ [L_i, U_i]  ∀i  (feature bounds)
    (3) x'_i ∈ {immutable features} = x_i  (protected attributes)
```

**Constraint Types:**

| Constraint Type | Description | Implementation |
|-----------------|-------------|----------------|
| Actionability | Only modifiable features can change | Feature-level actionability masks |
| Feasibility | Changes must be realistic | Domain-specific validity functions |
| Causality | Respect causal relationships | Causal graph constraints |
| Non-discrimination | No protected attribute changes | Immutable feature enforcement |
| Sparsity | Minimal number of changes | L₁ regularization term |

### 4.3 Component 3: Explanation Quality Auditor (EQA)

The EQA module provides automated assessment of explanation quality across multiple dimensions, enabling practitioners to evaluate and compare different XAI methods systematically.

**Evaluation Metrics:**

| Metric | Formula/Description | Target Range |
|--------|---------------------|--------------|
| Fidelity | Agreement between explainer and model predictions | > 0.90 |
| Stability | Consistency of explanations under small input perturbations | > 0.85 |
| Comprehensibility | Number of features needed for explanation | < 10 |
| Actionability Score | Proportion of actionable features in counterfactuals | > 0.80 |
| Discrimination Index | Fairness measure across protected groups | < 0.05 |
| Regulatory Compliance | Pass rate on compliance checklist | 100% |

---

## 5. Technical Implementation Details

### 5.1 Data Preprocessing Pipeline

The data preprocessing pipeline transforms raw transaction data into model-ready features while preserving information necessary for explanation generation.

**Stage 1: Temporal Feature Engineering**
- Extract hour, day of week, month, and cyclical representations (sin/cos) from `trans_date_trans_time`
- Calculate transaction velocity features (time since last transaction, transaction frequency in past hour/day)

**Stage 2: Geographic Feature Engineering**
- Compute Haversine distance between customer location (lat, long) and merchant location (merch_lat, merch_long)
- Create regional clustering features using K-means on geographic coordinates

**Stage 3: Categorical Encoding**
- Apply target encoding for high-cardinality features (merchant, job) with smoothing
- Use one-hot encoding for low-cardinality features (category, gender)
- Implement category aggregation features

**Stage 4: Numerical Feature Processing**
- Apply log transformation to skewed features (amt, city_pop)
- Use robust scaling (median, IQR) to handle outliers
- Create amount percentiles relative to user history

**Stage 5: Demographic Feature Engineering**
- Calculate age from date of birth
- Create income proxy features from city population
- Implement behavioral profiling features

### 5.2 Model Architecture Specifications

| Model | Architecture Details | Parameters | Training Config |
|-------|---------------------|------------|-----------------|
| XGBoost | max_depth=8, n_estimators=500, learning_rate=0.05 | ~2.5M | Early stopping, 5-fold CV |
| LightGBM | num_leaves=64, n_estimators=500, learning_rate=0.03 | ~1.8M | Class weights, 5-fold CV |
| Neural Network | 3 hidden layers [256, 128, 64], ReLU, Dropout 0.3 | ~150K | Adam, lr=0.001, batch=1024 |
| TabNet | n_d=32, n_a=32, n_steps=5, gamma=1.5 | ~500K | Adam, lr=0.02, virtual batches |
| Logistic Regression | L2 regularization, C=0.1 | ~200 | Class weights, saga solver |

### 5.3 XAI Implementation Specifications

| Component | Library | Custom Modifications | Computational Requirements |
|-----------|---------|---------------------|---------------------------|
| SHAP | shap 0.42+ | Stratified background sampling | GPU recommended for DeepExplainer |
| LIME | lime 0.2+ | Class-weighted sampling | CPU sufficient |
| Counterfactuals | dice-ml 0.8+ | Regulatory constraints | Optimization can be parallelized |
| Integrated Gradients | captum 0.7+ | Baseline distribution sampling | GPU required for efficiency |
| Attention Analysis | pytorch 2.0+ | Custom attention extraction | GPU required |

---

## 6. Experimental Design

### 6.1 Dataset Configuration

| Split | Samples | Fraud Rate | Purpose |
|-------|---------|------------|---------|
| Training | 389,803 (70%) | 0.38% | Model training with cross-validation |
| Validation | 83,530 (15%) | 0.38% | Hyperparameter tuning, early stopping |
| Test | 83,528 (15%) | 0.38% | Final evaluation, held-out |

**Imbalance Ratio Experiments:**
For robustness testing, additional experiments will use stratified sampling to create controlled imbalance ratios:
- 1:10
- 1:50
- 1:100
- 1:250 (original)
- 1:500

### 6.2 Baseline Models and Methods

| Category | Baselines | Rationale |
|----------|-----------|-----------|
| Detection Models | XGBoost, LightGBM, MLP, TabNet, Logistic Regression | Cover interpretability spectrum |
| XAI Methods | SHAP, LIME, Integrated Gradients, Attention | Standard comparison points |
| Counterfactual Methods | DiCE, FACE, Wachter | Regulatory compliance baselines |
| Imbalanced Learning | SMOTE, ADASYN, Class Weights, Focal Loss | Data-level vs algorithm-level |

### 6.3 Evaluation Metrics

#### 6.3.1 Detection Performance Metrics

| Metric | Formula | Why Important |
|--------|---------|---------------|
| AUC-ROC | TPR integration over FPR | Threshold-independent assessment |
| AUC-PRC | Precision integration over Recall | Appropriate for imbalanced data |
| F1-Score (Fraud) | 2×Precision×Recall/(P+R) | Balance precision and recall |
| Detection Rate @ 1% FPR | TPR at fixed FPR=0.01 | Operational constraint simulation |
| Matthews Correlation Coefficient | (TP×TN-FP×FN)/√(...) | Balanced measure for imbalance |

#### 6.3.2 Explanation Quality Metrics

| Metric | Definition | Measurement Method |
|--------|------------|-------------------|
| Fidelity | Explainer-model agreement | Prediction difference correlation |
| Stability | Explanation consistency | Jaccard similarity under perturbation |
| Comprehensibility | Explanation simplicity | Number of significant features |
| Correctness | Feature importance accuracy | Ground truth feature comparison |
| Actionability | Feasible counterfactual ratio | Domain expert evaluation |

### 6.4 Ablation Study Design

1. **IC-SHAP Ablation:** Compare standard SHAP vs. stratified background only vs. calibrated weights only vs. full IC-SHAP. Measure improvement in explanation stability and correctness for fraud instances.

2. **RC-CF Ablation:** Evaluate counterfactual quality with/without each constraint type (actionability, feasibility, causality, non-discrimination). Measure sparsity-achievability trade-offs.

3. **Model Complexity Impact:** Analyze how explanation quality varies with model complexity across the model spectrum from Logistic Regression to Neural Networks.

---

## 7. Python Implementation Templates

### 7.1 IC-SHAP Implementation Template

```python
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

class ImbalanceCalibratedSHAP:
    """
    Imbalance-Calibrated SHAP for extreme class imbalance scenarios.
    
    This implementation addresses the systematic bias in traditional SHAP
    when applied to imbalanced datasets by using stratified background
    sampling and calibration weights.
    """
    
    def __init__(self, model, X_train, y_train, n_background_samples=500):
        """
        Initialize IC-SHAP explainer.
        
        Parameters:
        -----------
        model : object
            Trained model with predict_proba method
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training labels (0: legitimate, 1: fraud)
        n_background_samples : int
            Number of samples per class for background distribution
        """
        self.model = model
        self.n_background_samples = n_background_samples
        
        # Create stratified background samples
        X_legit = X_train[y_train == 0]
        X_fraud = X_train[y_train == 1]
        
        # Sample with replacement for minority class
        self.bg_legit = X_legit.sample(
            n=min(n_background_samples, len(X_legit)), 
            random_state=42
        )
        self.bg_fraud = X_fraud.sample(
            n=min(n_background_samples, len(X_fraud)),
            replace=len(X_fraud) < n_background_samples,
            random_state=42
        )
        
        # Calculate calibration weights
        n_legit = (y_train == 0).sum()
        n_fraud = (y_train == 1).sum()
        self.w_legit = n_fraud / (n_legit + n_fraud)
        self.w_fraud = n_legit / (n_legit + n_fraud)
        
        print(f"Background samples - Legitimate: {len(self.bg_legit)}, Fraud: {len(self.bg_fraud)}")
        print(f"Calibration weights - w_legit: {self.w_legit:.4f}, w_fraud: {self.w_fraud:.4f}")
    
    def explain(self, X_explain, return_both=False):
        """
        Generate IC-SHAP explanations for given instances.
        
        Parameters:
        -----------
        X_explain : pd.DataFrame or np.ndarray
            Instances to explain
        return_both : bool
            If True, return both class-conditional SHAP values
            
        Returns:
        --------
        shap_values_ic : np.ndarray
            Imbalance-calibrated SHAP values
        """
        # Initialize explainers with different backgrounds
        if hasattr(self.model, 'predict_proba'):
            # For tree-based models
            try:
                explainer_legit = shap.TreeExplainer(self.model, self.bg_legit)
                explainer_fraud = shap.TreeExplainer(self.model, self.bg_fraud)
            except:
                # Fallback to KernelExplainer
                explainer_legit = shap.KernelExplainer(
                    self.model.predict_proba, self.bg_legit
                )
                explainer_fraud = shap.KernelExplainer(
                    self.model.predict_proba, self.bg_fraud
                )
        
        # Compute class-conditional SHAP values
        shap_legit = explainer_legit.shap_values(X_explain)
        shap_fraud = explainer_fraud.shap_values(X_explain)
        
        # Handle binary classification output format
        if isinstance(shap_legit, list):
            shap_legit = shap_legit[1]  # Take fraud class
            shap_fraud = shap_fraud[1]
        
        # Weighted aggregation
        shap_values_ic = (
            self.w_legit * shap_legit + 
            self.w_fraud * shap_fraud
        )
        
        if return_both:
            return shap_values_ic, shap_legit, shap_fraud
        
        return shap_values_ic
    
    def explain_instance(self, x_single):
        """
        Explain a single instance with detailed output.
        
        Parameters:
        -----------
        x_single : pd.Series or np.ndarray
            Single instance to explain
            
        Returns:
        --------
        dict : Contains SHAP values and feature rankings
        """
        if isinstance(x_single, pd.Series):
            X_explain = x_single.to_frame().T
            feature_names = x_single.index.tolist()
        else:
            X_explain = x_single.reshape(1, -1)
            feature_names = [f"feature_{i}" for i in range(len(x_single))]
        
        shap_values = self.explain(X_explain)[0]
        
        # Create feature importance ranking
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'top_5_features': importance_df.head(5)['feature'].tolist()
        }
```

### 7.2 RC-CF Implementation Template

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional

class RegulatoryCompliantCF:
    """
    Regulatory-Compliant Counterfactual Generator for Financial Fraud Detection.
    
    Generates counterfactual explanations that satisfy:
    - Actionability constraints (only modifiable features can change)
    - Feasibility constraints (changes must be realistic)
    - Non-discrimination constraints (protected attributes immutable)
    - Sparsity constraints (minimal number of changes)
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
            Features that cannot be changed (e.g., age, gender)
        bounds : Dict[str, Tuple[float, float]]
            Min-max bounds for each feature
        actionable_features : List[str], optional
            Features that can be modified (defaults to all non-immutable)
        """
        self.model = model
        self.feature_names = feature_names
        self.immutable_features = immutable_features
        self.bounds = bounds
        
        # Determine actionable features
        if actionable_features is None:
            self.actionable_features = [
                f for f in feature_names if f not in immutable_features
            ]
        else:
            self.actionable_features = actionable_features
        
        # Create index mappings
        self.immutable_indices = [
            feature_names.index(f) for f in immutable_features
        ]
        self.actionable_indices = [
            feature_names.index(f) for f in self.actionable_features
        ]
        
        # Optimization hyperparameters
        self.lambda_dist = 0.1      # Distance penalty weight
        self.lambda_sparse = 0.5    # Sparsity penalty weight
        self.lambda_actionable = 0.3  # Actionability penalty weight
    
    def generate_counterfactual(
        self,
        x_original: np.ndarray,
        target_class: int = 0,  # 0 = legitimate
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
            Number of candidate counterfactuals to generate
            
        Returns:
        --------
        dict : Contains best counterfactual and metadata
        """
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
        
        # Select best counterfactual (minimum changes)
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
        """Run single optimization with random initialization."""
        
        def objective(x_cf):
            """Combined objective function."""
            # Prediction loss: encourage class flip
            pred = self.model.predict_proba(x_cf.reshape(1, -1))[0, 1]
            pred_loss = max(0, pred - (1 - threshold)) if target_class == 0 else max(0, threshold - pred)
            
            # Distance loss: L2 distance from original
            dist_loss = np.linalg.norm(x_cf - x_original)
            
            # Sparsity loss: number of changed features
            sparse_loss = np.sum(np.abs(x_cf - x_original) > 1e-6)
            
            # Actionability loss: penalize changes to non-actionable features
            actionable_mask = np.zeros(len(x_original))
            actionable_mask[self.actionable_indices] = 1
            action_loss = np.sum((1 - actionable_mask) * np.abs(x_cf - x_original))
            
            return (
                pred_loss + 
                self.lambda_dist * dist_loss + 
                self.lambda_sparse * sparse_loss +
                self.lambda_actionable * action_loss
            )
        
        # Set up bounds
        opt_bounds = [
            self.bounds.get(self.feature_names[i], (None, None))
            for i in range(len(x_original))
        ]
        
        # Set up immutability constraints
        constraints = []
        for idx in self.immutable_indices:
            constraints.append({
                'type': 'eq',
                'fun': lambda x, idx=idx: x[idx] - x_original[idx]
            })
        
        # Random perturbation for diversity
        np.random.seed(seed)
        x_init = x_original.copy()
        for idx in self.actionable_indices:
            lb, ub = opt_bounds[idx]
            if lb is not None and ub is not None:
                x_init[idx] += np.random.uniform(-0.1, 0.1) * (ub - lb)
        
        # Optimize
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
        
        return {
            'success': result.success,
            'counterfactual': x_cf,
            'original': x_original,
            'n_changes': np.sum(changes),
            'changed_features': [
                self.feature_names[i] for i in range(len(changes)) if changes[i]
            ],
            'change_magnitudes': {
                self.feature_names[i]: x_cf[i] - x_original[i]
                for i in range(len(changes)) if changes[i]
            },
            'prediction_original': self.model.predict_proba(x_original.reshape(1, -1))[0, 1],
            'prediction_cf': self.model.predict_proba(x_cf.reshape(1, -1))[0, 1]
        }
    
    def batch_generate(
        self,
        X: np.ndarray,
        target_class: int = 0,
        **kwargs
    ) -> List[Dict]:
        """Generate counterfactuals for multiple instances."""
        results = []
        for i, x in enumerate(X):
            cf_result = self.generate_counterfactual(x, target_class, **kwargs)
            cf_result['instance_idx'] = i
            results.append(cf_result)
        return results
```

### 7.3 Complete Experiment Pipeline Template

```python
"""
Complete Experiment Pipeline for XAI Fraud Detection Research
=============================================================

This template provides the full experimental workflow from data loading
to result generation. Adapt as needed for specific experimental configurations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================

def load_and_preprocess_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the Kaggle Credit Card Fraud dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    X : pd.DataFrame
        Preprocessed features
    y : pd.Series
        Target labels
    """
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
    
    # ============ Temporal Features ============
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # ============ Geographic Features ============
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points."""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    df['distance_to_merchant'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    # ============ Amount Features ============
    df['amt_log'] = np.log1p(df['amt'])
    df['amt_zscore'] = (df['amt'] - df['amt'].mean()) / df['amt'].std()
    
    # ============ Categorical Encoding ============
    # Target encoding for high-cardinality features
    for col in ['merchant', 'job']:
        target_mean = df.groupby(col)['is_fraud'].mean()
        df[f'{col}_target_enc'] = df[col].map(target_mean)
    
    # One-hot encoding for low-cardinality
    df = pd.get_dummies(df, columns=['category', 'gender'], drop_first=True)
    
    # ============ Age Feature ============
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    
    # ============ Feature Selection ============
    feature_cols = [
        'amt', 'amt_log', 'amt_zscore',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month',
        'distance_to_merchant', 'age', 'city_pop',
        'merchant_target_enc', 'job_target_enc'
    ]
    
    # Add one-hot encoded columns
    feature_cols.extend([c for c in df.columns if c.startswith('category_')])
    feature_cols.extend([c for c in df.columns if c.startswith('gender_')])
    
    X = df[feature_cols].copy()
    y = df['is_fraud'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"\nProcessed feature shape: {X.shape}")
    print(f"Final fraud rate: {y.mean():.4%}")
    
    return X, y


# ============================================================
# SECTION 2: MODEL TRAINING
# ============================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier with class imbalance handling."""
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        max_depth=8,
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='auc'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM classifier with class imbalance handling."""
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        num_leaves=64,
        n_estimators=500,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    return model


# ============================================================
# SECTION 3: EVALUATION METRICS
# ============================================================

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Comprehensive model evaluation for fraud detection.
    
    Returns dict with all relevant metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'auc_prc': average_precision_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'detection_rate': tp / (tp + fn),  # Recall
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
    })
    
    # Detection rate at 1% FPR
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    fpr = fp / (fp + tn)
    # Find threshold for ~1% FPR
    for thresh in np.linspace(0, 1, 1000):
        y_pred_temp = (y_pred_proba >= thresh).astype(int)
        tn_temp, fp_temp, fn_temp, tp_temp = confusion_matrix(y_test, y_pred_temp).ravel()
        fpr_temp = fp_temp / (fp_temp + tn_temp)
        if fpr_temp <= 0.01:
            metrics['detection_rate_at_1pct_fpr'] = tp_temp / (tp_temp + fn_temp)
            break
    
    return metrics


# ============================================================
# SECTION 4: XAI EVALUATION
# ============================================================

def evaluate_explanation_stability(explainer, X_sample, n_perturbations=10, 
                                   perturbation_scale=0.01):
    """
    Evaluate explanation stability under small perturbations.
    
    Returns stability score (Jaccard similarity of top features).
    """
    base_shap = explainer.shap_values(X_sample[:1])[0]
    base_top_features = set(np.argsort(np.abs(base_shap))[-10:])
    
    jaccard_scores = []
    
    for _ in range(n_perturbations):
        # Add small noise
        X_perturbed = X_sample[:1] + np.random.normal(
            0, perturbation_scale, X_sample[:1].shape
        )
        perturbed_shap = explainer.shap_values(X_perturbed)[0]
        perturbed_top_features = set(np.argsort(np.abs(perturbed_shap))[-10:])
        
        # Jaccard similarity
        intersection = len(base_top_features & perturbed_top_features)
        union = len(base_top_features | perturbed_top_features)
        jaccard = intersection / union if union > 0 else 1.0
        jaccard_scores.append(jaccard)
    
    return np.mean(jaccard_scores)


def evaluate_explanation_fidelity(model, explainer, X_sample):
    """
    Evaluate explanation fidelity by measuring correlation between
    explainer predictions and model predictions.
    """
    shap_values = explainer.shap_values(X_sample)
    
    # Reconstruct predictions from SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Binary classification
    
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1]
    
    reconstructed_pred = base_value + shap_values.sum(axis=1)
    model_pred = model.predict_proba(X_sample)[:, 1]
    
    # Correlation between reconstructed and actual predictions
    correlation = np.corrcoef(reconstructed_pred, model_pred)[0, 1]
    
    return correlation


# ============================================================
# SECTION 5: MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiments(X, y, n_splits=5):
    """
    Run complete experimental pipeline with cross-validation.
    """
    results = {
        'xgboost': [],
        'lightgbm': [],
        'ic_shap_metrics': [],
        'standard_shap_metrics': []
    }
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Further split train for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
        )
        
        # Train models
        print("Training XGBoost...")
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        
        print("Training LightGBM...")
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        lgb_metrics = evaluate_model(lgb_model, X_test, y_test)
        
        results['xgboost'].append(xgb_metrics)
        results['lightgbm'].append(lgb_metrics)
        
        print(f"\nXGBoost AUC-ROC: {xgb_metrics['auc_roc']:.4f}")
        print(f"LightGBM AUC-ROC: {lgb_metrics['auc_roc']:.4f}")
        
        # XAI Evaluation (on subset for efficiency)
        X_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
        
        # Standard SHAP
        print("\nEvaluating Standard SHAP...")
        standard_explainer = shap.TreeExplainer(xgb_model)
        standard_stability = evaluate_explanation_stability(
            standard_explainer, X_test_sample.values
        )
        standard_fidelity = evaluate_explanation_fidelity(
            xgb_model, standard_explainer, X_test_sample.values
        )
        
        # IC-SHAP
        print("Evaluating IC-SHAP...")
        ic_shap = ImbalanceCalibratedSHAP(
            xgb_model, X_train.values, y_train.values
        )
        ic_shap_values, _, _ = ic_shap.explain(X_test_sample.values, return_both=True)
        
        results['standard_shap_metrics'].append({
            'stability': standard_stability,
            'fidelity': standard_fidelity
        })
        
        print(f"Standard SHAP - Stability: {standard_stability:.4f}, Fidelity: {standard_fidelity:.4f}")
    
    return results


# ============================================================
# SECTION 6: RESULTS VISUALIZATION
# ============================================================

def plot_results(results):
    """Generate publication-ready visualization of results."""
    
    # Detection performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # AUC-ROC comparison
    xgb_aucs = [r['auc_roc'] for r in results['xgboost']]
    lgb_aucs = [r['auc_roc'] for r in results['lightgbm']]
    
    axes[0].boxplot([xgb_aucs, lgb_aucs], labels=['XGBoost', 'LightGBM'])
    axes[0].set_ylabel('AUC-ROC', fontsize=12)
    axes[0].set_title('Detection Performance Comparison', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # AUC-PRC comparison
    xgb_prcs = [r['auc_prc'] for r in results['xgboost']]
    lgb_prcs = [r['auc_prc'] for r in results['lightgbm']]
    
    axes[1].boxplot([xgb_prcs, lgb_prcs], labels=['XGBoost', 'LightGBM'])
    axes[1].set_ylabel('AUC-PRC', fontsize=12)
    axes[1].set_title('Precision-Recall Performance', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detection_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # XAI Quality comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stability_scores = [r['stability'] for r in results['standard_shap_metrics']]
    fidelity_scores = [r['fidelity'] for r in results['standard_shap_metrics']]
    
    x = np.arange(2)
    means = [np.mean(stability_scores), np.mean(fidelity_scores)]
    stds = [np.std(stability_scores), np.std(fidelity_scores)]
    
    ax.bar(x, means, yerr=stds, capsize=5, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Stability', 'Fidelity'])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('XAI Quality Metrics', fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('xai_quality.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Load data (replace with actual file path)
    # X, y = load_and_preprocess_data('path/to/fraud_data.csv')
    
    # Run experiments
    # results = run_experiments(X, y)
    
    # Plot results
    # plot_results(results)
    
    print("Pipeline template loaded successfully.")
    print("Uncomment the main execution block to run experiments.")
```

---

## 8. Expected Contributions

### 8.1 Theoretical Contributions

1. **Formal analysis of XAI method behavior under extreme class imbalance**, establishing theoretical foundations for explanation bias in imbalanced scenarios.

2. **Mathematical framework for Imbalance-Calibrated SHAP (IC-SHAP)** with convergence guarantees and bias correction properties.

3. **Regulatory compliance theory** linking counterfactual explanations to financial regulations (GDPR, ECOA, Fair Credit Reporting Act).

### 8.2 Methodological Contributions

1. **Novel IC-SHAP algorithm** with stratified background sampling and calibration weights for reliable explanations in imbalanced settings.

2. **Regulatory-Compliant Counterfactual Generator (RC-CF)** with actionability, feasibility, and non-discrimination constraints.

3. **Comprehensive Explanation Quality Auditor (EQA) framework** with metrics spanning fidelity, stability, comprehensibility, and compliance.

4. **Open-source implementation** (Python package) for reproducibility and adoption by practitioners and researchers.

### 8.3 Empirical Contributions

- Extensive experiments on Kaggle Credit Card Fraud dataset (556,861 transactions, 0.38% fraud rate) establishing benchmarks for XAI in fraud detection.

- Comparative analysis of 5 detection models and 4 XAI methods under varying imbalance ratios.

- User study with financial professionals evaluating explanation utility for fraud investigation decisions.

---

## 9. Target Paper Structure

| Section | Content | Approximate Length |
|---------|---------|-------------------|
| Abstract | Problem, methods, key findings, contributions | 250 words |
| Introduction | Motivation, gap, objectives, paper organization | 1,500 words |
| Related Work | Fraud detection, XAI, imbalanced learning, regulations | 2,500 words |
| Proposed Method | IC-SHAP, RC-CF, EQA framework details | 3,000 words |
| Experimental Setup | Dataset, baselines, metrics, implementation | 1,500 words |
| Results & Analysis | Detection performance, explanation quality | 2,500 words |
| Discussion | Implications, limitations, future work | 1,000 words |
| Conclusion | Summary, contributions, impact | 500 words |

**Total Target Length:** ~12,500 words

---

## 10. Research Timeline

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1: Foundation | Literature review, data preparation, baseline models | Weeks 1-3 | Review document, clean dataset, baseline results |
| Phase 2: Method Development | IC-SHAP, RC-CF, EQA implementation | Weeks 4-6 | Working code, unit tests |
| Phase 3: Experiments | Main experiments, ablation studies | Weeks 7-9 | Results tables, figures |
| Phase 4: Analysis | Statistical analysis, result interpretation | Weeks 10-11 | Analysis report |
| Phase 5: Writing | Draft writing, revision, formatting | Weeks 12-14 | Complete draft |
| Phase 6: Submission | Final review, submission preparation | Weeks 15-16 | Submitted manuscript |

---

## 11. Required Dependencies and Resources

### 11.1 Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21 | Numerical computations |
| pandas | >=1.5 | Data manipulation |
| scikit-learn | >=1.2 | Machine learning utilities |
| xgboost | >=1.7 | Gradient boosting models |
| lightgbm | >=3.3 | Light gradient boosting |
| shap | >=0.42 | SHAP explanations |
| lime | >=0.2 | LIME explanations |
| dice-ml | >=0.8 | Counterfactual generation |
| torch | >=2.0 | Neural network models |
| captum | >=0.7 | Integrated gradients |
| matplotlib | >=3.6 | Visualization |
| seaborn | >=0.12 | Statistical visualization |

**Installation Command:**
```bash
pip install numpy pandas scikit-learn xgboost lightgbm shap lime dice-ml torch captum matplotlib seaborn
```

### 11.2 Computational Resources

- **GPU:** 16GB+ VRAM recommended for deep learning models and SHAP DeepExplainer
- **RAM:** 32GB+ system RAM for large-scale experiments
- **Storage:** 100GB+ for dataset, models, and experimental artifacts
- **Processing:** Parallel processing capability recommended for hyperparameter tuning

---

## 12. Target Journal Specifications

| Journal | Impact Factor | Acceptance Rate | Review Time | Focus |
|---------|---------------|-----------------|-------------|-------|
| Financial Innovation | 8.4 (Q1) | ~15% | 2-4 months | Financial technology, AI |
| Expert Systems with Applications | 8.5 (Q1) | ~18% | 3-5 months | Applied AI, expert systems |
| IEEE TNNLS | 10.4 (Q1) | ~12% | 4-6 months | Neural networks, learning |
| Knowledge-Based Systems | 8.8 (Q1) | ~16% | 2-4 months | AI systems, applications |
| Pattern Recognition | 8.0 (Q1) | ~18% | 3-5 months | Pattern analysis, ML |

---

## 13. Instructions for AI Execution

This document provides complete specifications for executing the research project. When handing to an AI model, provide the following additional context:

### Dataset Access
Download the Kaggle Credit Card Fraud dataset from:
```
https://www.kaggle.com/datasets/chetanmittal033/credit-card-fraud-data/data
```

### Execution Guidelines

1. **Environment Setup:** Execute all code in a Python 3.9+ environment with all dependencies installed.

2. **Figure Generation:** Generate publication-ready figures using the matplotlib/seaborn specifications provided in the templates.

3. **Writing Standards:** Follow academic writing standards for all documentation. Use formal language, cite relevant literature, and maintain objectivity.

4. **Reproducibility:** Ensure reproducibility by:
   - Setting random seeds (RANDOM_STATE = 42)
   - Documenting all hyperparameters
   - Providing complete code for all experiments
   - Including version numbers for all dependencies

5. **Paper Formatting:** Follow the target journal's formatting guidelines. Most Q1 journals use either:
   - IEEE format (double-column)
   - Elsevier format (single-column)
   - Springer format

### Expected Outputs

The AI model should produce:

1. **Complete Python Code:** All experimental code with documentation
2. **Experimental Results:** Tables and figures for all experiments
3. **Analysis Report:** Statistical analysis and interpretation
4. **Paper Draft:** Complete manuscript following the target paper structure
5. **Supplementary Materials:** Code repository, additional figures, detailed results

---

**This blueprint is designed to be comprehensive and self-contained, enabling an AI model to execute the complete research workflow from data preprocessing through paper submission preparation.**
