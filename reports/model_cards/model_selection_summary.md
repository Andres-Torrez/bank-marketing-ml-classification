# Model Selection Summary

## Objective
Compare all trained candidate models and select the best model for the next phase of the project.

## Evaluation Criteria
The models were compared using:

- ROC-AUC (primary metric)
- F1-score
- Recall
- Precision
- Overfitting gap

Overfitting control rule:

gap = train_roc_auc - cv_roc_auc

A model is considered stable if:

gap <= 0.05

## Candidate Models
- DummyClassifier
- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- HistGradientBoostingClassifier

## Selection Logic
The final candidate should:

1. Achieve strong ROC-AUC on test data
2. Show controlled overfitting
3. Maintain acceptable recall and F1-score
4. Be computationally reasonable
5. Be justifiable from a business and modeling perspective

## Final Decision
[Write the selected model here]

## Justification
[Explain why it was selected]

## Notes
- `duration` was excluded from the production modeling pipeline due to leakage risk.
- The target class is imbalanced, so ROC-AUC was prioritized over accuracy.

# Model Selection Summary

## Objective
Compare all trained candidate models and select the best model for the next phase of the project.

## Evaluation Criteria
The models were compared using:

- ROC-AUC (primary metric)
- F1-score
- Recall
- Precision
- Overfitting gap

Overfitting control rule:

gap = train_roc_auc - cv_roc_auc

A model is considered stable if:

gap <= 0.05

## Candidate Models
- DummyClassifier
- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- HistGradientBoostingClassifier

## Final Decision
The selected model is: **GradientBoostingClassifier**

## Justification
Although HistGradientBoostingClassifier achieved the highest ROC-AUC, it exceeded the acceptable overfitting threshold (`gap_roc_auc > 0.05`).

GradientBoostingClassifier achieved the best balance between predictive performance and generalization:

- strong ROC-AUC on cross-validation and test
- controlled overfitting gap
- stable behavior across folds
- better production reliability under the project constraint

For this reason, GradientBoostingClassifier was selected as the final model for the application phase.

## Key Metrics
- CV ROC-AUC: 0.7921
- Test ROC-AUC: 0.8019
- Overfitting gap: 0.0147
- Test F1: 0.3104
- Test Recall: 0.2023
- Test Precision: 0.6667

## Notes
- `duration` was excluded from the production pipeline due to leakage risk.
- The dataset is imbalanced, therefore ROC-AUC was prioritized over accuracy.