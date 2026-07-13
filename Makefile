.PHONY: run-all test

run-all:
	python labs/01_ml_foundations/run_experiment.py
	python labs/02_deep_learning/run_experiment.py
	python labs/03_nlp_ir_rag/run_experiment.py
	python labs/04_knowledge_graphs/run_experiment.py
	python labs/05_agents_rl/run_experiment.py
	python labs/06_xai_trustworthy_ai/run_experiment.py
	python labs/07_mlops_reproducibility/run_experiment.py
	python labs/08_financial_ai/run_experiment.py

test:
	python -m pytest -q
