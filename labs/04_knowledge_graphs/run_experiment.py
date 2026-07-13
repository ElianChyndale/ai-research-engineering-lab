from __future__ import annotations

from pathlib import Path
import sys

import networkx as nx

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lab_utils import ensure_results_dir, token_set, write_csv, write_jsonl


TRIPLES = [
    ("SolarCo", "issues", "GreenBondA"),
    ("GreenBondA", "finances", "SolarFarm"),
    ("SolarFarm", "located_in", "Munster"),
    ("SolarCo", "has_risk", "WaterDisclosureRisk"),
    ("AuditorOne", "audits", "SolarCo"),
    ("GreenBondA", "has_metric", "AvoidedEmissions"),
]

DOCS = [
    {"doc_id": "E1", "entity": "SolarCo", "text": "SolarCo issued GreenBondA with audited avoided emissions."},
    {"doc_id": "E2", "entity": "WaterDisclosureRisk", "text": "Water disclosure remains a risk in supplier evidence."},
    {"doc_id": "E3", "entity": "SolarFarm", "text": "SolarFarm is financed by GreenBondA in Munster."},
]

QUESTIONS = [
    {"question_id": "K1", "text": "Who audits SolarCo and what bond did it issue", "expected": "E1", "entity": "SolarCo"},
    {"question_id": "K2", "text": "What risk is linked to SolarCo", "expected": "E2", "entity": "WaterDisclosureRisk"},
    {"question_id": "K3", "text": "Which project is financed by GreenBondA", "expected": "E3", "entity": "SolarFarm"},
]


def main() -> None:
    results = ensure_results_dir()
    graph = nx.MultiDiGraph()
    graph.add_edges_from((src, dst, {"relation": rel}) for src, rel, dst in TRIPLES)
    write_jsonl(
        results / "kg_triples.jsonl",
        [{"subject": src, "predicate": rel, "object": dst} for src, rel, dst in TRIPLES],
    )
    rows = []
    for question in QUESTIONS:
        q_tokens = token_set(question["text"])
        base_scores = [len(q_tokens & token_set(doc["text"])) for doc in DOCS]
        for method in ["vector_only", "kg_assisted"]:
            scores = list(base_scores)
            if method == "kg_assisted":
                for idx, doc in enumerate(DOCS):
                    if doc["entity"] == question["entity"] or nx.has_path(graph.to_undirected(), question["entity"], doc["entity"]):
                        scores[idx] += 2
            best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
            top_doc = DOCS[best_idx]["doc_id"]
            rows.append(
                {
                    "question_id": question["question_id"],
                    "method": method,
                    "top_doc": top_doc,
                    "hit_score": 1.0 if top_doc == question["expected"] else 0.0,
                    "graph_nodes": graph.number_of_nodes(),
                    "graph_edges": graph.number_of_edges(),
                }
            )
    write_csv(results / "kg_rag_comparison.csv", rows)


if __name__ == "__main__":
    main()
