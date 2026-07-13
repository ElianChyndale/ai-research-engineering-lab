from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lab_utils import ensure_results_dir, write_csv


SEED = 42


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    results = ensure_results_dir()
    x, y = make_classification(
        n_samples=260,
        n_features=10,
        n_informative=6,
        random_state=SEED,
        class_sep=1.1,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=SEED, stratify=y
    )
    rows = []
    curves = {}
    for learning_rate in [0.001, 0.01]:
        model = MLPClassifier(
            hidden_layer_sizes=(16,),
            learning_rate_init=learning_rate,
            max_iter=140,
            random_state=SEED,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append(
            {
                "model": "sklearn_mlp",
                "ablation": f"learning_rate_{learning_rate}",
                "accuracy": round(float(accuracy_score(y_test, pred)), 6),
                "f1": round(float(f1_score(y_test, pred)), 6),
                "final_loss": round(float(model.loss_), 6),
            }
        )
        curves[str(learning_rate)] = model.loss_curve_
    write_csv(results / "deep_learning_ablation.csv", rows)
    plt.figure(figsize=(4, 3))
    for label, curve in curves.items():
        plt.plot(curve, label=f"lr={label}")
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results / "training_curve.svg")


if __name__ == "__main__":
    main()
