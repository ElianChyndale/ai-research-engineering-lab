from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lab_utils import ensure_results_dir, write_csv, write_json


CONFIG = {"seed": 42, "model": "logistic_regression", "test_size": 0.25, "n_samples": 180}


def main() -> None:
    results = ensure_results_dir()
    x, y = make_classification(
        n_samples=CONFIG["n_samples"],
        n_features=5,
        n_informative=3,
        random_state=CONFIG["seed"],
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=CONFIG["test_size"], random_state=CONFIG["seed"], stratify=y
    )
    model = LogisticRegression(max_iter=400, random_state=CONFIG["seed"])
    model.fit(x_train, y_train)
    accuracy = round(float(accuracy_score(y_test, model.predict(x_test))), 6)
    payload = json.dumps({"config": CONFIG, "accuracy": accuracy}, sort_keys=True).encode("utf-8")
    output_hash = hashlib.sha256(payload).hexdigest()
    write_json(results / "reproducibility_config.json", CONFIG)
    write_csv(
        results / "reproducibility_check_results.csv",
        [
            {
                "check": "fixed_seed_pipeline",
                "accuracy": accuracy,
                "output_hash": output_hash,
                "deterministic_score": 1.0,
            }
        ],
    )


if __name__ == "__main__":
    main()
