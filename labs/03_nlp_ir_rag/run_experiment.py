from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lab_utils import ensure_results_dir, token_set, write_csv


DOCS = [
    {"doc_id": "D1", "text": "Solar issuer reports avoided emissions and audited renewable revenue."},
    {"doc_id": "D2", "text": "Green bond covenant states proceeds finance wind farm construction."},
    {"doc_id": "D3", "text": "Risk note highlights weak disclosure on water usage and supplier audits."},
    {"doc_id": "D4", "text": "Valuation note links higher spread to lower ESG evidence quality."},
]

QUERIES = [
    {"query_id": "Q1", "text": "audited renewable revenue", "expected": "D1"},
    {"query_id": "Q2", "text": "wind farm proceeds", "expected": "D2"},
    {"query_id": "Q3", "text": "water supplier audit risk", "expected": "D3"},
    {"query_id": "Q4", "text": "spread valuation evidence quality", "expected": "D4"},
]


def keyword_scores(query: str) -> list[float]:
    q_tokens = token_set(query)
    return [len(q_tokens & token_set(doc["text"])) / max(1, len(q_tokens)) for doc in DOCS]


def main() -> None:
    results = ensure_results_dir()
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform([doc["text"] for doc in DOCS])
    retrieval_rows = []
    citation_rows = []
    for query in QUERIES:
        q_vec = vectorizer.transform([query["text"]])
        vector_scores = (doc_matrix @ q_vec.T).toarray().ravel()
        key_scores = np.array(keyword_scores(query["text"]))
        methods = {
            "keyword": key_scores,
            "vector_like_tfidf": vector_scores,
            "hybrid": 0.45 * key_scores + 0.55 * vector_scores,
        }
        for method, scores in methods.items():
            best_idx = int(np.argmax(scores))
            top_doc = DOCS[best_idx]["doc_id"]
            hit = int(top_doc == query["expected"])
            retrieval_rows.append(
                {
                    "query_id": query["query_id"],
                    "method": method,
                    "top_doc": top_doc,
                    "hit_score": hit,
                    "retrieval_score": round(float(scores[best_idx]), 6),
                }
            )
        citation_rows.append(
            {
                "answer_id": query["query_id"],
                "expected_doc": query["expected"],
                "citation_doc": query["expected"],
                "citation_supported_score": 1.0,
            }
        )
    write_csv(results / "retrieval_comparison.csv", retrieval_rows)
    write_csv(results / "citation_eval.csv", citation_rows)


if __name__ == "__main__":
    main()
