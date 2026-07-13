# Oral Exam Questions

For each question, provide:
1. A definition
2. An equation
3. A code pointer (file:line)
4. A failure example
5. A project connection

---

## Linear Regression

1. What is ordinary least squares? When does it fail?
2. Derive the normal equations from the least-squares objective.
3. What is the difference between OLS and gradient descent for linear regression?
4. When would gradient descent outperform OLS?
5. How does linear regression connect to EcoQuant's factor models?

## Logistic Regression

1. What is the sigmoid function and why is it used for classification?
2. Write the binary cross-entropy loss function.
3. What is L2 regularization and how does it affect the coefficients?
4. When does logistic regression fail to converge?
5. How does logistic regression connect to calibration in this lab?

## PCA

1. What does PCA optimize?
2. Derive the principal components from the covariance matrix.
3. What is explained variance ratio?
4. When should you use SVD instead of eigendecomposition?
5. How does PCA connect to dimensionality reduction in retrieval?

## BM25

1. What are the three components of the BM25 score?
2. Write the BM25 scoring equation.
3. What is the role of k1 and b parameters?
4. When does BM25 reduce to simple TF-IDF?
5. How does BM25 connect to EcoQuant's retrieval system?

## Train/Calibration/Test Separation

1. Why is a three-way split needed?
2. What happens if you tune hyperparameters on the test set?
3. What is the role of a calibration set?
4. How do you detect if separation has been violated?
5. How does this lab enforce separation?

## Data Leakage

1. Define data leakage in machine learning.
2. Give three concrete examples of leakage.
3. How does normalization leakage occur?
4. What is the difference between leakage and overfitting?
5. How does the leakage checklist in this lab help?

## Brier Score

1. What does the Brier score measure?
2. Write the Brier score formula.
3. What is a perfect Brier score? What is the worst?
4. How does Brier score differ from accuracy?
5. How does Brier score connect to calibration?

## Calibration

1. What does it mean for a model to be calibrated?
2. What is Expected Calibration Error (ECE)?
3. What is a reliability diagram?
4. How does calibration differ from discrimination?
5. How does the calibration module in this lab work?

## Retrieval Metrics

1. What is precision at k?
2. What is recall at k?
3. What is MAP (Mean Average Precision)?
4. What is NDCG?
5. How does BM25 ranking connect to these metrics?

## Reproducibility

1. What makes an experiment reproducible?
2. What is the role of seeds in reproducibility?
3. What is an experiment manifest?
4. What is an artifact hash and why is it important?
5. How does this lab enforce reproducibility?

## Artifact Hashes

1. What is SHA-256?
2. Why hash experiment artifacts?
3. What does a hash mismatch indicate?
4. How does the validation script check artifacts?
5. What are the limitations of hash-based validation?

## Baselines and Ablations

1. What is a baseline in ML research?
2. What is an ablation study?
3. Why report all ablation results, not just favorable ones?
4. What is a negative control?
5. How does this lab's checklist help?
