# Data Leakage Checklist

## Future Information

- [ ] Do any features use data from after the prediction point?
- [ ] Are timestamps correctly aligned?
- [ ] Is there look-ahead in feature engineering?

## Train/Test Contamination

- [ ] Are train and test sets truly disjoint?
- [ ] Do any test samples appear in training data?
- [ ] Are data splits done before any preprocessing?

## Normalization Leakage

- [ ] Is the scaler fit only on training data?
- [ ] Are statistics (mean, std, min, max) computed before the split?
- [ ] Is any global statistics used in per-sample normalization?

## Threshold-Selection Leakage

- [ ] Is the decision threshold selected on the test set?
- [ ] Is threshold tuning done on a held-out calibration set?

## Evaluator-Label Access

- [ ] Does the evaluation pipeline see labels during metric computation only?
- [ ] Is there any human-in-the-loop that has seen test labels?

## Duplicate Entities

- [ ] Are the same entities (users, documents, events) in both train and test?
- [ ] Is there entity-level grouping in the split?

## Repeated Documents

- [ ] Are near-duplicate documents in different splits?
- [ ] Is text deduplication done before splitting?

## Claim Selection After Seeing Results

- [ ] Are you reporting only the experiments that worked?
- [ ] Is the hypothesis stated before running the experiment?
- [ ] Are all ablation results reported, not just favorable ones?
