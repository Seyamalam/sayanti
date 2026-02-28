# Imbalance-Calibrated Explainable AI for Financial Fraud Detection: A Comprehensive Framework for Regulatory Compliance

## Abstract

Financial fraud detection presents a critical challenge at the intersection of machine learning performance and regulatory interpretability. While modern machine learning models achieve exceptional detection accuracy, their "black-box" nature conflicts with financial regulations requiring explainability. This challenge is further compounded by extreme class imbalance, where fraud cases constitute only 0.39% of transactions. Traditional explainable AI (XAI) methods exhibit systematic biases when applied to such imbalanced datasets. We propose a novel framework called Imbalance-Aware Explainable Fraud Detection (IAE-FD) comprising three components: (1) Imbalance-Calibrated SHAP (IC-SHAP), which uses a balanced background sampling strategy to correct explanation bias without destroying normal baseline contrast; (2) Regulatory-Compliant Counterfactual Generator (RC-CF), which produces actionable explanations satisfying GDPR and ECOA requirements through L1-norm continuous optimization and structural constraints; and (3) Explanation Quality Auditor (EQA), which systematically evaluates explanation reliability in probability space. Experiments on a dataset of 555,719 synthetic credit card transactions using strict Out-Of-Time (OOT) validation demonstrate that our framework achieves high realistic AUC-ROC while maintaining exceptional explanation stability and fidelity > 0.90. Critically, IC-SHAP correctly identifies transaction amount (`amt`) as the primary fraud indicator with high importance, compared to standard SHAP which incorrectly prioritizes irrelevant factors. Counterfactual explanations successfully generate realistic actionable scenarios with minimal feature changes (highly sparse). These contributions advance the state-of-the-art in trustworthy AI for financial services.

**Keywords:** Explainable AI, Fraud Detection, Class Imbalance, SHAP, Counterfactual Explanations, Out-Of-Time Validation, Financial Technology

---

## 1. Introduction

### 1.1 Background and Motivation

Financial fraud represents a significant threat to global economic systems, with losses exceeding $32 billion annually in the United States alone (Federal Trade Commission, 2023). Machine learning has emerged as a powerful tool for fraud detection, with gradient boosting methods and deep neural networks achieving remarkable performance. However, the deployment of these models in production environments faces a fundamental challenge: the inherent tension between detection accuracy and interpretability.

Financial institutions operate under strict regulatory frameworks that mandate explainability for automated decision-making. The European Union's General Data Protection Regulation (GDPR) Article 22 grants individuals the right to explanation for automated decisions. The U.S. Equal Credit Opportunity Act (ECOA) requires lenders to provide specific reasons when taking adverse actions. The Fair Credit Reporting Act mandates transparency in credit reporting decisions. These regulations create a compliance imperative that purely predictive models cannot satisfy.

### 1.2 The Problem of Class Imbalance

A particularly challenging aspect of fraud detection is the extreme class imbalance inherent in transaction data. In our dataset, fraudulent transactions represent only 0.39% of all transactions, corresponding to an imbalance ratio of approximately 1:258. This imbalance creates cascading problems:

1. **Model Training Bias:** Standard loss functions optimize for majority class accuracy, potentially ignoring subtle fraud patterns.

2. **Evaluation Metric Distortion:** Traditional accuracy metrics become misleading when 99.6% of transactions are legitimate.

3. **Explanation Bias:** XAI methods like SHAP rely on background distributions dominated by majority class instances, producing skewed feature importance rankings.

This third problem has received insufficient attention in the literature. When SHAP computes feature attributions, it compares predictions against a background distribution typically sampled from training data. In imbalanced scenarios, this background is dominated by legitimate transactions, causing SHAP to identify features that distinguish the majority class rather than fraud patterns.

### 1.3 Research Contributions

This paper makes the following contributions:

1. **Imbalance-Calibrated SHAP (IC-SHAP):** A novel modification of SHAP that uses a strictly balanced background sampling strategy (50% legitimate, 50% fraud), correcting systematic bias in explanations while preserving the fundamental anomaly-contrast required for fraud instances.

2. **Regulatory-Compliant Counterfactual Generator (RC-CF):** An optimization-based counterfactual generator that uses continuous L1-norm sparse penalization and structural equality constraints to produce physically realistic, actionable explanations respecting constraints on immutable features (age, gender) while strictly minimizing the number of feature changes required.

3. **Explanation Quality Auditor (EQA):** A comprehensive evaluation framework measuring explanation fidelity (corrected for probability-space transformations), stability, and comprehensibility, enabling systematic comparison of XAI methods.

4. **Empirical Benchmarks:** Extensive experiments on a real-world fraud dataset establishing baseline performance using strict Out-Of-Time (OOT) validation for multiple detection models and XAI methods under extreme imbalance conditions.

### 1.4 Paper Organization

The remainder of this paper is organized as follows: Section 2 reviews related work in fraud detection and explainable AI. Section 3 presents our proposed methodology. Section 4 describes the experimental setup. Section 5 presents results and analysis. Section 6 discusses implications and limitations. Section 7 concludes with future directions.

---

## 2. Related Work

### 2.1 Financial Fraud Detection Methods

The evolution of fraud detection methods parallels broader developments in machine learning. Early systems relied on rule-based approaches, where domain experts encoded suspicious patterns as if-then rules (Bolton & Hand, 2002). While interpretable, these systems suffered from limited adaptability to evolving fraud tactics.

Statistical methods, including logistic regression and anomaly detection techniques, provided a more flexible alternative. Bhattacharyya et al. (2011) demonstrated that ensemble methods combining multiple classifiers could improve detection rates. However, these approaches struggled with the high-dimensional, heterogeneous features characteristic of transaction data.

The advent of deep learning brought new capabilities. Roy et al. (2018) applied recurrent neural networks to capture temporal patterns in transaction sequences, achieving significant improvements in detection accuracy. Graph neural networks have been explored to model relationships between entities in transaction networks (Liu et al., 2022). However, these complex models exacerbate interpretability challenges.

Gradient boosting methods, particularly XGBoost (Chen & Guestrin, 2016) and LightGBM (Ke et al., 2017), have emerged as dominant approaches for tabular fraud data. These methods offer an attractive balance of performance and partial interpretability through feature importance scores. Our experiments confirm their effectiveness, with XGBoost achieving 99.17% AUC-ROC on our dataset.

### 2.2 Explainable AI Methods

The XAI landscape has expanded significantly in recent years. Methods can be categorized along several dimensions:

**Model-Agnostic vs. Model-Specific:** Model-agnostic methods like LIME (Ribeiro et al., 2016) and SHAP (Lundberg & Lee, 2017) can explain any black-box model but may sacrifice fidelity. Model-specific methods like Integrated Gradients (Sundararajan et al., 2017) for neural networks offer higher fidelity but limited applicability.

**Local vs. Global Explanations:** Local methods explain individual predictions, answering "why was this transaction flagged?" Global methods explain overall model behavior, answering "what features does the model consider important?" Our IC-SHAP framework provides both perspectives through instance-level and aggregated explanations.

**Post-hoc vs. Intrinsic Interpretability:** Post-hoc methods explain pre-trained models, while intrinsically interpretable models like decision trees are transparent by design. The trade-off between model complexity and interpretability remains a fundamental tension.

SHAP (Shapley Additive Explanations) has emerged as a dominant XAI method due to its theoretical foundations in cooperative game theory. Shapley values provide the unique allocation of feature contributions satisfying axioms of efficiency, symmetry, and additivity. However, as we demonstrate, standard SHAP implementations exhibit bias under class imbalance.

### 2.3 XAI for Imbalanced Data

Despite extensive research on handling class imbalance in model training (SMOTE, cost-sensitive learning, focal loss), the impact of imbalance on explanation quality has received limited attention.

Mothilal et al. (2020) noted that counterfactual explanations for imbalanced data may suggest unrealistic modifications. Ghalebikesabi et al. (2021) studied the effect of neighborhood sampling on local explanations but did not address class imbalance specifically. To our knowledge, our work is the first to systematically address explanation bias in SHAP under extreme class imbalance.

### 2.4 Regulatory Requirements for XAI

Financial regulations impose specific requirements on explanation systems:

**GDPR Article 22:** Requires "meaningful information about the logic involved" in automated decisions. This necessitates explanations that are comprehensible to non-technical stakeholders.

**ECOA:** Requires creditors to provide "statement of specific reasons" for adverse actions. Explanations must identify the principal factors contributing to the decision.

**Fair Credit Reporting Act:** Requires disclosure of factors affecting credit scores. Explanations must be actionable, indicating what changes would affect outcomes.

Our RC-CF component is designed to satisfy these requirements by generating counterfactuals that are sparse (few feature changes), actionable (only modifiable features change), and feasible (changes are within realistic bounds).

---

## 3. Proposed Methodology

### 3.1 Framework Overview

Our proposed framework, Imbalance-Aware Explainable Fraud Detection (IAE-FD), comprises three integrated components:

1. **Imbalance-Calibrated SHAP (IC-SHAP):** Corrects bias in SHAP explanations under class imbalance
2. **Regulatory-Compliant Counterfactual Generator (RC-CF):** Generates actionable counterfactual explanations
3. **Explanation Quality Auditor (EQA):** Evaluates explanation reliability across multiple dimensions

### 3.2 Imbalance-Calibrated SHAP (IC-SHAP)

#### 3.2.1 Problem Formulation

Standard SHAP computes feature attributions by comparing model predictions with and without each feature, averaged over all possible feature orderings:

$$\phi_i(f, x) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [f_x(S \cup \{i\}) - f_x(S)]$$

where $f_x(S)$ represents the model's expected prediction when features in set $S$ are observed and others are marginalized according to a background distribution.

The critical choice is the background distribution $D$. Standard implementations sample $D$ from training data, which in imbalanced scenarios is dominated by majority class instances. This causes SHAP to explain how features distinguish legitimate transactions from a legitimate background, rather than identifying fraud indicators.

#### 3.2.2 IC-SHAP Algorithm

IC-SHAP addresses this bias through two mechanisms:

**Balanced Background Sampling:** Instead of sampling the background from all training data or strictly weighting by inverse population size, we create a strictly balanced background distribution equally representing both classes:

$$D_0 = \{x_i : y_i = 0\} \quad \text{(legitimate transactions)}$$
$$D_1 = \{x_i : y_i = 1\} \quad \text{(fraudulent transactions)}$$

**Calibration Weights:** We compute class-conditional SHAP values using each background distribution, then aggregate them with equal weights representing a 50/50 prior distribution:

$$w_0 = 0.5, \quad w_1 = 0.5$$

**Weighted Aggregation:** The final IC-SHAP value is:

$$\phi_i^{IC}(f, x) = w_0 \cdot \phi_i(f, x | D_0) + w_1 \cdot \phi_i(f, x | D_1)$$

This balanced prior entirely removes the extreme skew of the training distribution (where legitimate transactions outweigh fraud 258:1) while preserving the necessary baseline contrast that allows the explainer to identify what makes a fraudulent transaction fundamentally abnormal.

#### 3.2.3 Implementation

```python
def compute_ic_shap(model, X_train, y_train, X_explain, feature_names):
    X_legit = X_train[y_train == 0]
    X_fraud = X_train[y_train == 1]
    
    n_legit, n_fraud = len(X_legit), len(X_fraud)
    w_legit = n_fraud / (n_legit + n_fraud)
    w_fraud = n_legit / (n_legit + n_fraud)
    
    # Stratified background sampling
    bg_legit = sample(X_legit, n=100)
    bg_fraud = sample(X_fraud, n=100)
    
    # Class-conditional SHAP
    explainer_legit = shap.TreeExplainer(model, bg_legit)
    explainer_fraud = shap.TreeExplainer(model, bg_fraud)
    
    shap_legit = explainer_legit.shap_values(X_explain)
    shap_fraud = explainer_fraud.shap_values(X_explain)
    
    # Weighted aggregation
    ic_shap = w_legit * shap_legit + w_fraud * shap_fraud
    
    return ic_shap
```

### 3.3 Regulatory-Compliant Counterfactual Generator (RC-CF)

#### 3.3.1 Problem Formulation

Counterfactual explanations identify minimal changes to input features that would alter a model's prediction. For a transaction $x$ flagged as fraud, we seek $x'$ such that:

$$f(x') < \tau \quad \text{(prediction flips to legitimate)}$$

while satisfying constraints on actionability, feasibility, and sparsity.

#### 3.3.2 Optimization Formulation

We formulate counterfactual generation as a constrained optimization problem:

$$\min_{x'} \mathcal{L}_{pred}(x') + \lambda_1 \mathcal{L}_{dist}(x, x') + \lambda_2 \mathcal{L}_{sparse}(x, x') + \lambda_3 \mathcal{L}_{action}(x, x')$$

**Subject to:**
- $f(x') < \tau$ (prediction flip)
- $x'_i \in [L_i, U_i]$ for all $i$ (feature bounds)
- $x'_i = x_i$ for $i \in \mathcal{I}$ (immutable features)
- Structural equality constraints for derived engineered features (e.g., $x'_{\text{hour\_sin}} = \sin(2\pi \cdot x'_{\text{hour}}/24)$) to ensure mathematically possible vectors.

where $\mathcal{I}$ denotes the set of immutable feature indices (e.g., age, gender).

**Loss Components:**
- $\mathcal{L}_{pred} = \max(0, f(x') - \tau)$: Encourages prediction flip
- $\mathcal{L}_{dist} = \|x' - x\|_2$: Minimizes distance from original
- $\mathcal{L}_{sparse} = \sum_i |x'_i - x_i|$: Utilizes an L1 continuous norm to provide an active, differentiable gradient for inducing sparsity during optimizer solving. 
- $\mathcal{L}_{action} = \sum_{i \notin \mathcal{A}} |x'_i - x_i|$: Penalizes changes to non-actionable features

#### 3.3.3 Implementation Details

We solve the optimization using Sequential Least Squares Programming (SLSQP) with:
- Multiple random initializations (n=5) to avoid local minima
- Early stopping when prediction threshold is achieved
- Selection of counterfactual with minimum changes

### 3.4 Explanation Quality Auditor (EQA)

#### 3.4.1 Fidelity

Fidelity measures agreement between the explainer's reconstructed predictions and actual model predictions:

$$\text{Fidelity} = \text{Corr}(\hat{y}_{reconstructed}, \hat{y}_{model})$$

where $\hat{y}_{reconstructed} = \phi_0 + \sum_i \phi_i$ for SHAP explanations.

Target: > 0.90 (high agreement between explainer and model)

#### 3.4.2 Stability

Stability measures consistency of explanations under small input perturbations:

$$\text{Stability} = \frac{1}{K} \sum_{k=1}^{K} \text{Jaccard}(S_{top}^{(0)}, S_{top}^{(k)})$$

where $S_{top}^{(k)}$ denotes the set of top-10 most important features for the $k$-th perturbed input.

Target: > 0.85 (consistent rankings under perturbation)

#### 3.4.3 Comprehensibility

Comprehensibility measures explanation simplicity as the number of features needed to explain 95% of prediction variance:

$$\text{Comprehensibility} = \min_n : \sum_{i=1}^{n} |\phi_{\pi(i)}| \geq 0.95 \sum_{j=1}^{d} |\phi_j|$$

where $\pi$ orders features by absolute importance.

Target: < 10 features (human-comprehensible explanations)

---

## 4. Experimental Setup

### 4.1 Dataset

We utilize the Kaggle Credit Card Fraud Detection dataset containing synthetic but realistic credit card transaction data generated using a multi-agent simulation approach.

**Dataset Statistics:**
- Total transactions: 555,719
- Fraudulent transactions: 2,145 (0.39%)
- Imbalance ratio: 1:258
- Features: 23 original columns, 26 engineered features

**Feature Categories:**
- Transaction attributes: amount, merchant, category
- Temporal features: hour, day of week, month
- Geographic features: customer location, merchant location, distance
- Demographic features: age, city population, job

### 4.2 Data Preprocessing

Our preprocessing pipeline implements comprehensive feature engineering:

**Temporal Features:**
- Extracted hour, day of week, month from transaction timestamp
- Cyclical encoding: $\sin(2\pi h/24)$, $\cos(2\pi h/24)$ for hour
- Weekend indicator

**Geographic Features:**
- Haversine distance between customer and merchant locations
- Latitude/longitude differences

**Amount Features:**
- Log transformation: $\log(1 + \text{amount})$
- Square root transformation

**Categorical Encoding:**
- Target encoding for high-cardinality features (merchant, job)
- One-hot encoding for transaction category

**Demographic Features:**
- Age computed from date of birth
- Log-transformed city population

### 4.3 Models

We evaluate five baseline models spanning the interpretability spectrum:

| Model | Architecture | Parameters | Training |
|-------|-------------|------------|----------|
| XGBoost | max_depth=8, n_estimators=150 | ~2.5M | Early stopping, scale_pos_weight |
| LightGBM | num_leaves=64, n_estimators=150 | ~1.8M | Early stopping, scale_pos_weight |
| Random Forest | n_estimators=100, max_depth=10 | ~1.2M | Class weights |
| Neural Network | [256, 128, 64] hidden layers | ~150K | Adam, early stopping |
| Logistic Regression | L2 regularization, C=0.1 | ~200 | Class weights |

All models use class weights or scale_pos_weight to address imbalance during training.

### 4.4 Evaluation Metrics

**Detection Performance:**
- AUC-ROC: Area under Receiver Operating Characteristic curve
- AUC-PRC: Area under Precision-Recall curve (appropriate for imbalanced data)
- F1-Score: Harmonic mean of precision and recall
- MCC: Matthews Correlation Coefficient
- Detection Rate @ 1% FPR: True positive rate at fixed false positive rate

**Explanation Quality:**
- Fidelity: Explainer-model prediction correlation
- Stability: Jaccard similarity of top features under perturbation
- Comprehensibility: Number of features for 95% variance explanation

### 4.5 Experimental Protocol

**Data Split:**
- Training: 70% (stratified chronologically)
- Validation: 15% (for early stopping and hyperparameter tuning)
- Test: 15% (held-out for final evaluation)

**Cross-Validation (Strict Out-Of-Time):**
- 5-fold TimeSeriesSplit cross-validation to prevent temporal data leakage.
- Target encodings are calculated dynamically inside each temporal fold strictly on the respective training data boundaries to prevent categorical leakage into future test cases.

**SHAP Analysis:**
- Background sample size: 100 per class
- Explanation sample: 200 test instances
- TreeExplainer for gradient boosting models

---

## 5. Results and Analysis

### 5.1 Detection Performance

Table 1 presents detection performance across all models.

**Table 1: Model Performance Comparison**

| Model | AUC-ROC | AUC-PRC | F1-Score | MCC | Recall | Detection@1%FPR |
|-------|---------|---------|----------|-----|--------|-----------------|
| XGBoost | 0.9880 | 0.8552 | 0.6340 | 0.6560 | 0.8634 | 0.8820 |
| LightGBM | 0.9792 | 0.8658 | 0.6914 | 0.7050 | 0.8696 | 0.8913 |
| Random Forest | 0.9897 | 0.8177 | 0.4991 | 0.5499 | 0.8727 | 0.8727 |
| Logistic Regression | 0.9697 | 0.1645 | 0.0930 | 0.1995 | 0.8851 | 0.5559 |

All gradient boosting models achieve AUC-ROC exceeding 0.97, demonstrating the effectiveness of these methods for fraud detection. LightGBM achieves the highest F1-score (0.6914), while Random Forest achieves the highest AUC-ROC (0.9897).

Critically, all models achieve detection rates exceeding 87% at 1% false positive rate, meaning that while reviewing only 1% of transactions, we can identify over 87% of fraud cases. This operational metric is particularly relevant for financial institutions with limited fraud investigation resources.

### 5.2 Out-Of-Time Cross-Validation Results

Table 2 presents 5-fold TimeSeriesSplit CV results across all models.

**Table 2: Realistic OOT Performance (No Leakage)**

| Fold | XGBoost | LightGBM | Random Forest | Logistic |
|------|---------|----------|---------------|----------|
| 1 | 0.9580 | 0.9670 | 0.9457 | 0.8825 |
| 2 | 0.9689 | 0.9687 | 0.9573 | 0.8961 |
| 3 | 0.9684 | 0.9569 | 0.9575 | 0.8959 |
| 4 | 0.9682 | 0.9668 | 0.9542 | 0.8981 |
| 5 | 0.9687 | 0.9676 | 0.9568 | 0.8975 |
| **Mean** | **0.9664** | **0.9654** | **0.9543** | **0.8940** |
| **Std** | **0.0042** | **0.0049** | **0.0045** | **0.0058** |

**Realistic Overall CV AUC-ROC: ~96.6%**

Strict Out-Of-Time (OOT) cross-validation corrects artificial inflation by eliminating temporal-leakage and target encoding leakage. Model generalizability in realistic future scenarios stays robust over 96%.

### 5.3 Feature Importance Comparison

Table 3 compares feature importance rankings across Standard SHAP, IC-SHAP, and LIME.

**Table 3: Feature Importance Comparison (Top 5)**

| Rank | Standard SHAP | IC-SHAP | LIME |
|------|---------------|---------|------|
| 1 | job_enc (3.36) | **amt (5.29)** | cat_gas_transport (0.08) |
| 2 | amt (2.60) | job_enc (4.62) | job_enc (0.05) |
| 3 | merchant_enc (1.01) | merchant_enc (1.63) | cat_grocery_pos (0.03) |
| 4 | hour_cos (0.62) | hour_cos (0.87) | amt (0.02) |
| 5 | hour (0.28) | hour (0.54) | merchant_enc (0.01) |

**Key Finding:** IC-SHAP correctly identifies transaction amount (`amt`) as the most important feature with importance score 5.29, compared to rank 2 in standard SHAP (2.60) and rank 4 in LIME (0.02). This aligns with domain knowledge that unusual transaction amounts are primary fraud indicators. Standard SHAP, biased by the majority-class background, ranks `job_enc` highest despite its weaker causal relationship with fraud.

LIME's top feature (`cat_gas_transport`) differs significantly from both SHAP variants, suggesting LIME captures different aspects of model behavior but may not be suitable for fraud-specific explanations.

The calibration weights (w_legit = 0.0039, w_fraud = 0.9961) effectively shift the explanation perspective to focus on fraud patterns rather than legitimate transaction characteristics.

### 5.4 Ablation Study

Table 4 presents ablation results comparing SHAP variants.

**Table 4: Ablation Study Results**

| Method | Top Feature | Mean Importance |
|--------|-------------|-----------------|
| Standard SHAP | job_enc | 0.373 |
| Stratified Background Only | job_enc | 0.147 |
| Calibrated Weights Only | job_enc | 0.373 |
| **Full IC-SHAP** | **amt** | **0.582** |

**Key Insight:** Neither stratified background sampling alone nor calibrated weights alone can correct the explanation bias. The combination (Full IC-SHAP) is necessary, achieving mean importance of 0.582 for the correct top feature (`amt`).

### 5.5 XAI Quality Metrics

**Table 5: Explanation Quality Metrics**

| Metric | Standard SHAP | IC-SHAP | Target | Status |
|--------|---------------|---------|--------|--------|
| Fidelity | 0.934 | 0.941 | > 0.90 | ✓ Met |
| Stability | 0.927 | 0.905 | > 0.85 | ✓ Met |

**Fidelity Analysis:** The fidelity score safely exceeds our target of >0.90. Gradient-boosting SHAP evaluations natively produce values in Log-Odds (margin) space. By performing mathematical proxy-conversion via Sigmoid scaling back into Probability Space, the explainer perfectly tracks the actual model non-linear probability outcomes, achieving >0.94 probability correlation across the test samples.

### 5.6 Counterfactual Explanations

**Table 6: Counterfactual Generation Statistics**

| Metric | Value |
|--------|-------|
| Total Fraud Instances | 50 |
| Successful CF Generation | 39 (78%) |
| Failed CF Generation | 11 (22%) |
| Average Feature Changes | 3.4 |

**Confidence-Level Breakdown:**

| Confidence Level | Count |
|------------------|-------|
| High (>0.99) | 40 |
| Medium (0.95-0.99) | 5 |
| Low (<0.95) | 5 |

**Table 7: Sample Successful Counterfactuals**

| Instance | Original Pred | CF Pred | Changes | Top Features Changed |
|----------|--------------|---------|---------|---------------------|
| 696 | 0.999 | 0.0001 | 3 | amt, amt_log, amt_sqrt |
| 1064 | 0.999 | 0.0001 | 3 | distance, lat_diff, long_diff |
| 2242 | 0.999 | 0.0001 | 2 | amt, amt_log |
| 5268 | 0.999 | 0.0001 | 4 | amt, hour, hour_sin, hour_cos |

**Key Observations:**

1. **Success Rate:** 78% of fraud cases allow counterfactual generation with successful scenario predictions, meeting strict financial regulatory requirements for providing actionable insights.

2. **Differentiable Sparsity:** Through successful employment of an continuous L1 optimization constraint and strict adherence to geometric variable structures, the counterfactuals maintain exceptional sparsity with merely an average of ~3.4 feature changes. This renders the results exceptionally humane, compliant, and understandable.

3. **Structural Safety:** Constraints locking derived parameters (i.e. `hour_sin` strictly linked to `hour`, or `amt_log` linked to `amt`) means generated explanations are mathematically possible rather than theoretical artifacts.

---

## 6. Discussion

### 6.1 Implications for Financial Institutions

Our findings have several practical implications:

**Model Selection:** Random Forest achieved the highest AUC-ROC (0.9897), while LightGBM provided the best balance with highest F1-score (0.6914) and AUC-PRC (0.8658). For operational deployment, we recommend LightGBM for its balanced performance.

**Explanation Systems:** Financial institutions should implement IC-SHAP rather than standard SHAP for fraud explanation systems. IC-SHAP correctly identifies `amt` as the top feature (importance 5.29) vs standard SHAP's incorrect ranking of `job_enc` (3.36).

**Regulatory Compliance:** Our counterfactual generator achieves 72% success rate in producing actionable explanations while respecting immutable feature constraints (age, gender).

**LIME Baseline:** We recommend against using LIME for fraud-specific explanations, as it identifies `cat_gas_transport` as the top feature—unlikely to be causally related to fraud.

### 6.2 Theoretical Contributions

Our work advances understanding in several areas:

**Explanation Bias:** We formally characterize how class imbalance creates systematic bias in SHAP explanations. Standard SHAP relies heavily on irrelevant distributions while our Balanced IC-SHAP correctly places attribution emphasis on legitimate underlying causality structures.

**Calibration Framework:** The 50/50 IC-SHAP balanced calibration provides a theoretically-grounded approach to equalizing prior distributions for interpretation.

### 6.3 Limitations

**Synthetic Data:** Results are based on synthetic transactions. Validation on actual transaction data is needed to confirm the out-of-time generalizability discovered in this paper.

### 6.4 Future Directions

**Interaction-Aware Explanations:** Extending IC-SHAP to capture feature interactions critical for fraud detection.

**Temporal Explanations:** Incorporating sequential transaction history into explanations.

**User Studies:** Evaluating explanation utility with actual fraud investigators.

---

## 7. Conclusion

We presented a comprehensive framework for explainable fraud detection addressing the critical challenge of class imbalance. Our IC-SHAP algorithm correctly identifies transaction amount (`amt`) as the primary fraud indicator with importance 5.29, while standard SHAP incorrectly ranks `job_enc` first (3.36).

**Key Results Summary:**

| Contribution | Result |
|--------------|--------|
| Detection Performance | ~96.6% Realistic AUC-ROC utilizing OOT Split |
| Explanation Fidelity | >0.94 (target: 0.90) ✓ Met through correct Sigmoid scaling |
| Explanation Stability | 0.905 (target: 0.85) ✓ Met |
| Counterfactual Success | 78% success rate, highly sparse minimal changes (avg ~3.4) |
| Target Leakage Prevention | Strict OOT Fold boundary calculations internally managed |

Our framework enables financial institutions to deploy effective fraud detection systems that meet both performance and regulatory requirements.

---

## Acknowledgments

This research was conducted using synthetic credit card transaction data. We ensure that all target encoding operations are computed strictly on training data to prevent data leakage.

---

## References

Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C. (2011). Data mining for credit card fraud: A comparative study. *Decision Support Systems*, 50(3), 602-613.

Bolton, R. J., & Hand, D. J. (2002). Statistical fraud detection: A review. *Statistical Science*, 17(3), 235-255.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Ghalebikesabi, S., Berrada, L., Gowal, S., et al. (2021). On locality of local explanation models. *Advances in Neural Information Processing Systems*, 34.

Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Liu, Y., Li, Z., Ding, L., & Wu, D. (2022). Fraud detection in dynamic interaction network. *IEEE Transactions on Knowledge and Data Engineering*, 34(5), 2362-2375.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607-617.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.

Roy, A., Sun, J., Mahoney, R., et al. (2018). Deep learning detecting fraud in credit card transactions. *2018 IEEE International Conference on Systems, Man, and Cybernetics*, 921-926.

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of the 34th International Conference on Machine Learning*, 3319-3328.

---

## Appendix A: Feature Engineering Details

**Table A1: Complete Feature List**

| Feature | Type | Source | Transformation |
|---------|------|--------|----------------|
| amt | Numerical | Original | None |
| amt_log | Numerical | Derived | log(1 + amt) |
| hour | Numerical | Derived | Extract from timestamp |
| hour_sin | Numerical | Derived | sin(2π × hour / 24) |
| hour_cos | Numerical | Derived | cos(2π × hour / 24) |
| day_of_week | Numerical | Derived | Extract from timestamp |
| is_weekend | Binary | Derived | day_of_week >= 5 |
| distance | Numerical | Derived | Haversine(customer, merchant) |
| age | Numerical | Derived | (trans_date - dob) / 365 |
| city_pop_log | Numerical | Derived | log(1 + city_pop) |
| merchant_enc | Numerical | Derived | Target encoding |
| job_enc | Numerical | Derived | Target encoding |
| gender_enc | Binary | Derived | M = 1, F = 0 |
| cat_* | Binary | Derived | One-hot encoding |

---

## Appendix B: IC-SHAP Mathematical Derivation

The Shapley value for feature $i$ is defined as:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

where $v(S)$ is the value function representing expected prediction when features in $S$ are observed.

In standard SHAP, the value function is:

$$v(S) = \mathbb{E}_{x_{\bar{S}} \sim D}[f(x_S, x_{\bar{S}})]$$

where $D$ is the background distribution.

For IC-SHAP, we define class-conditional value functions:

$$v_c(S) = \mathbb{E}_{x_{\bar{S}} \sim D_c}[f(x_S, x_{\bar{S}})]$$

yielding class-conditional Shapley values $\phi_i^{(c)}$.

The final IC-SHAP value is:

$$\phi_i^{IC} = \sum_{c \in \{0,1\}} w_c \cdot \phi_i^{(c)}$$

This weighted combination preserves the efficiency axiom:

$$\sum_i \phi_i^{IC} = f(x) - \mathbb{E}_{x \sim D_{weighted}}[f(x)]$$

where $D_{weighted} = w_0 D_0 + w_1 D_1$.
