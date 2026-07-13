from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lab_utils import ensure_results_dir, write_csv


SEED = 42


def main() -> None:
    results = ensure_results_dir()
    x, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=SEED,
        class_sep=1.2,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=SEED, stratify=y
    )
    model = LogisticRegression(max_iter=500, random_state=SEED)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=5, strategy="uniform")
    calibration_error = float(np.mean(np.abs(frac_pos - mean_pred)))
    write_csv(
        results / "ml_foundations_metrics.csv",
        [
            {
                "model": "logistic_regression",
                "accuracy": round(float(accuracy_score(y_test, pred)), 6),
                "f1": round(float(f1_score(y_test, pred)), 6),
                "calibration_error": round(calibration_error, 6),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
        ],
    )
    plt.figure(figsize=(4, 3))
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction positive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results / "calibration_curve.svg")


if __name__ == "__main__":
    main()
