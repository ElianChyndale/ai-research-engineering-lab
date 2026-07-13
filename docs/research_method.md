# Research Method Template

## Research Question

[What specific question are you trying to answer?]

## Hypothesis

[What do you expect to find, and why?]

## Baseline

[What is the simplest credible baseline for comparison?]

## Proposed Method

[What method are you testing, and how does it differ from the baseline?]

## Dataset

- Source: [synthetic / real / hybrid]
- Size: [n samples, d features]
- Train/calibration/test split: [ratios and method]
- Preprocessing: [steps applied]

## Train/Calibration/Test Boundary

- Training data: [what the model sees]
- Calibration data: [what tuning uses]
- Test data: [what evaluation uses — must be unseen]

## Leakage Risks

- [ ] Future information in features
- [ ] Train/test contamination
- [ ] Normalization computed on full data
- [ ] Threshold selected on test set
- [ ] Evaluator has access to labels
- [ ] Duplicate entities across splits

## Metrics

| Metric | Definition | Expected Range |
|--------|-----------|---------------|
| [name] | [formula] | [range] |

## Ablations

| Component Removed | Expected Effect |
|------------------|----------------|
| [component] | [what changes] |

## Failure Analysis

- What would falsify the hypothesis?
- What edge cases could break the method?

## Reproducibility

- Seed: [fixed seed]
- Config file: [path]
- Run command: [exact command]
- Artifact location: [path]

## Permitted Claims

Based on this experiment, I can claim:
- [ ] [specific finding with evidence]

I cannot claim:
- [ ] [what the experiment doesn't support]
