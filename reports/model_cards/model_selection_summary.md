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