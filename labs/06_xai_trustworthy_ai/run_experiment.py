from __future__ import annotations

from pathlib import Path
import sys

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lab_utils import ensure_results_dir, write_csv


SEED = 42


def main() -> None:
    results = ensure_results_dir()
    x, y = make_classification(
        n_samples=220,
        n_features=6,
        n_informative=4,
        random_state=SEED,
        class_sep=1.0,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=SEED, stratify=y
    )
    model = RandomForestClassifier(n_estimators=60, random_state=SEED, max_depth=4)
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    importance = permutation_importance(model, x_test, y_test, random_state=SEED, n_repeats=5)
    rows = [
        {
            "feature": f"risk_feature_{idx}",
            "importance_score": round(max(0.0, float(score)), 6),
            "model_accuracy": round(float(accuracy), 6),
        }
        for idx, score in enumerate(importance.importances_mean, start=1)
    ]
    write_csv(results / "xai_feature_importance.csv", rows)
    write_csv(
        results / "ai_risk_register.csv",
        [
            {
                "risk_id": "R1",
                "risk": "Synthetic data overstates readiness",
                "mitigation": "Label outputs as local benchmark results",
                "severity_score": 0.7,
            },
            {
                "risk_id": "R2",
                "risk": "Feature importance misread as causality",
                "mitigation": "Report it as model sensitivity only",
                "severity_score": 0.6,
            },
        ],
    )


if __name__ == "__main__":
    main()
