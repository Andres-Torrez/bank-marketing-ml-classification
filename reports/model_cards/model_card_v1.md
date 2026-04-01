# Model Card — Bank Marketing Term Deposit Prediction

## 1. Model Overview

**Project objective:**  
Predict whether a client will subscribe to a term deposit based on demographic and campaign-related information.

**Problem type:**  
Binary classification

**Target variable:**  
`y`
- `yes` → client subscribes
- `no` → client does not subscribe

**Final selected model:**  
`GradientBoostingClassifier`

---

## 2. Business Context

The goal of this model is to support campaign decision-making by estimating the likelihood that a client will subscribe to a term deposit.

A higher predicted probability may help prioritize contact strategies or identify more promising client segments.

---

## 3. Dataset

**Dataset used:**  
Bank Marketing Dataset

**Rows:** 45,211  
**Columns:** 17

**Class distribution:**
- no → ~88.3%
- yes → ~11.7%

This class imbalance makes accuracy an insufficient primary metric.

---

## 4. Data Processing Decisions

### 4.1 Leakage Control
The variable `duration` was excluded from the production model because it is only known after the phone call is completed.

Using it would introduce data leakage and make the model unsuitable for real-world pre-contact prediction.

### 4.2 Handling of Unknown Categories
Values such as `unknown` were treated as valid categories rather than missing values, since they represent real operational states in the dataset.

### 4.3 Train/Test Strategy
- Train/test split: 80/20
- Stratified split
- Random state: 42

### 4.4 Cross Validation
- Stratified K-Fold
- Number of folds: 5

---

## 5. Modeling Strategy

Several model families were evaluated:

- DummyClassifier
- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- HistGradientBoostingClassifier

The final model was selected based on:

- ROC-AUC (primary metric)
- overfitting control
- generalization stability
- test performance

---

## 6. Evaluation Metrics

### Primary Metric
- ROC-AUC

### Secondary Metrics
- F1-score
- Recall
- Precision
- Accuracy

### Overfitting Definition
Overfitting gap was defined as:

`gap = train_roc_auc - cv_roc_auc`

Acceptance criterion:

`gap <= 0.05`

---

## 7. Final Model Performance

**Selected model:** GradientBoostingClassifier

**Cross-validation ROC-AUC:** 0.7921  
**Test ROC-AUC:** 0.8019  
**Overfitting gap:** 0.0147  
**Test F1-score:** 0.3104  
**Test Recall:** 0.2023  
**Test Precision:** 0.6667  
**Test Accuracy:** 0.8948

### Interpretation
The selected model achieved the best balance between predictive performance and generalization.

Although HistGradientBoostingClassifier obtained a slightly higher ROC-AUC, it exceeded the acceptable overfitting threshold and was therefore not selected.

---

## 8. Baseline Comparison

The final model outperformed the baseline approaches:

- DummyClassifier: no predictive value beyond majority-class guessing
- LogisticRegression: stable but lower predictive power
- DecisionTreeClassifier: weaker generalization than ensemble methods
- RandomForestClassifier: competitive, but lower recall and ROC-AUC than the selected final model

---

## 9. Limitations

- The dataset is imbalanced, which makes minority-class detection more difficult.
- Recall remains moderate, meaning that some subscribing clients are still missed.
- The model was trained on historical campaign data and may degrade if client behavior changes over time.
- No threshold tuning was applied yet.
- No production monitoring or drift detection has been implemented at this stage.

---

## 10. Ethical / Operational Considerations

- The model should support decision-making, not replace human judgment.
- Predictions should not be interpreted as certainty.
- Campaign outcomes may change over time, so retraining and monitoring are necessary for long-term deployment.

---

## 11. Production Readiness

Current status:
- reproducible preprocessing pipeline
- leakage-controlled feature set
- saved final model
- Streamlit app for inference

Pending for stronger production readiness:
- model monitoring
- feedback collection
- automated retraining pipeline
- threshold optimization
- deployment and containerization

---

## 12. Conclusion

`GradientBoostingClassifier` was selected as the final production candidate because it provided the best trade-off between performance and stability under the project constraints.

The model is suitable for demonstration and portfolio-level deployment, with clear next steps for further improvement.