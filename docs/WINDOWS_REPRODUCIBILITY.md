# Windows Reproducibility

This repo does not require `make` on Windows. `make run-all` and `make test` are optional shortcuts for systems where Make is installed.

## Run All Labs

From the `ai-research-engineering-lab` directory:

```powershell
python labs/01_ml_foundations/run_experiment.py
python labs/02_deep_learning/run_experiment.py
python labs/03_nlp_ir_rag/run_experiment.py
python labs/04_knowledge_graphs/run_experiment.py
python labs/05_agents_rl/run_experiment.py
python labs/06_xai_trustworthy_ai/run_experiment.py
python labs/07_mlops_reproducibility/run_experiment.py
python labs/08_financial_ai/run_experiment.py
```

## Run Tests

```powershell
python -m pytest -q
```

## Expected Outputs

Each lab writes deterministic result files under:

```text
research/results/
```

The portfolio-level verification runner also checks the AI Lab:

```powershell
python portfolio-package/scripts/run_all_research_checks.py
powershell -ExecutionPolicy Bypass -File portfolio-package/scripts/run_all_research_checks.ps1
```

Run those portfolio commands from the workspace root.
