# Baseline and Ablation Checklist

## Baseline Requirements

- [ ] Simplest credible baseline implemented
- [ ] Same data used for baseline and proposed method
- [ ] Same evaluation cutoff for both
- [ ] Same evaluation metrics for both
- [ ] Same compute budget where relevant
- [ ] Baseline is not a straw man (use a real published method)

## Ablation Requirements

- [ ] Each component removed one at a time
- [ ] Results reported for all ablations, not just favorable ones
- [ ] Negative controls included (e.g., random features, shuffled labels)
- [ ] Interaction effects noted when multiple components are removed

## Common Mistakes

- Comparing against a weak baseline to inflate improvement
- Reporting only the ablation that shows the biggest drop
- Not controlling for compute (faster method with same accuracy is a win)
- Using different random seeds without reporting variance
