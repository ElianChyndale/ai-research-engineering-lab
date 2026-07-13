from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results"

LAB_OUTPUTS = {
    "01_ml_foundations": ["ml_foundations_metrics.csv"],
    "02_deep_learning": ["deep_learning_ablation.csv"],
    "03_nlp_ir_rag": ["retrieval_comparison.csv", "citation_eval.csv"],
    "04_knowledge_graphs": ["kg_triples.jsonl", "kg_rag_comparison.csv"],
    "05_agents_rl": ["rl_learning_curve.csv", "agent_task_eval.csv"],
    "06_xai_trustworthy_ai": ["xai_feature_importance.csv", "ai_risk_register.csv"],
    "07_mlops_reproducibility": ["reproducibility_check_results.csv"],
    "08_financial_ai": ["bond_pricing_tests.csv", "financial_risk_eval.csv"],
}


def run_lab(lab_name: str) -> None:
    script = ROOT / "labs" / lab_name / "run_experiment.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_csv_has_rows(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"{path.name} should contain at least one data row"
    for row in rows:
        for key, value in row.items():
            if key.endswith(("accuracy", "f1", "precision", "recall", "calibration_error", "score")):
                number = float(value)
                assert 0.0 <= number <= 1.0


def assert_jsonl_has_rows(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, f"{path.name} should contain at least one JSONL record"


def test_each_lab_generates_parseable_outputs() -> None:
    for lab_name, output_names in LAB_OUTPUTS.items():
        run_lab(lab_name)
        for output_name in output_names:
            output_path = RESULTS / output_name
            assert output_path.exists(), f"{output_name} was not generated"
            if output_path.suffix == ".csv":
                assert_csv_has_rows(output_path)
            elif output_path.suffix == ".jsonl":
                assert_jsonl_has_rows(output_path)


def test_lab_outputs_are_deterministic() -> None:
    tracked = [RESULTS / "ml_foundations_metrics.csv", RESULTS / "reproducibility_check_results.csv"]
    run_lab("01_ml_foundations")
    run_lab("07_mlops_reproducibility")
    first_hashes = {path.name: file_hash(path) for path in tracked}
    run_lab("01_ml_foundations")
    run_lab("07_mlops_reproducibility")
    second_hashes = {path.name: file_hash(path) for path in tracked}
    assert first_hashes == second_hashes
