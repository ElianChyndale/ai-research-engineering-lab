"""BM25 scorer from the published equation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from airelab.core.config import ExperimentConfig
from airelab.experiments.registry import register


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer."""
    return text.lower().split()


class BM25:
    """Educational BM25 implementation (Robertson et al.).

    Args:
        corpus: List of document strings.
        k1: Term frequency saturation parameter (default 1.2).
        b: Document length normalization parameter (default 0.75).
    """

    def __init__(
        self,
        corpus: list[str],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n_docs = len(corpus)

        self._doc_tokens: list[list[str]] = [_tokenize(doc) for doc in corpus]
        self._doc_lengths: list[int] = [len(t) for t in self._doc_tokens]
        self._avg_dl = sum(self._doc_lengths) / self.n_docs if self.n_docs > 0 else 0.0

        self._df: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            for token in set(tokens):
                self._df[token] += 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _score_doc(self, query_terms: list[str], doc_idx: int) -> float:
        doc_tokens = self._doc_tokens[doc_idx]
        doc_len = self._doc_lengths[doc_idx]
        tf = Counter(doc_tokens)

        score = 0.0
        for term in query_terms:
            term_freq = tf.get(term, 0)
            idf = self._idf(term)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * doc_len / self._avg_dl
            )
            score += idf * numerator / denominator
        return score

    def score(self, query: str) -> list[float]:
        query_terms = _tokenize(query)
        if not query_terms:
            return [0.0] * self.n_docs
        return [self._score_doc(query_terms, i) for i in range(self.n_docs)]

    def rank(self, query: str) -> list[int]:
        scores = self.score(query)
        return sorted(range(self.n_docs), key=lambda i: (-scores[i], i))


@register("bm25")
def run_bm25(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Run a miniature BM25 retrieval experiment."""
    params = config.parameters
    k1 = params.get("k1", 1.2)
    b = params.get("b", 0.75)

    # Synthetic corpus
    corpus = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with multiple layers",
        "natural language processing handles text and speech data",
        "information retrieval finds relevant documents for queries",
        "reinforcement learning trains agents through rewards",
        "supervised learning uses labeled training data",
        "unsupervised learning discovers hidden patterns in data",
        "calibration measures how well predicted probabilities match outcomes",
    ]

    queries = ["machine learning", "neural networks", "document retrieval", "probability calibration"]

    bm25 = BM25(corpus, k1=k1, b=b)

    # Score and rank for each query
    rankings_path = run_dir / "rankings.csv"
    with rankings_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "rank", "doc_id", "score", "document"])
        for query in queries:
            scores = bm25.score(query)
            ranked = bm25.rank(query)
            for rank_pos, doc_idx in enumerate(ranked):
                writer.writerow([
                    query,
                    rank_pos + 1,
                    doc_idx,
                    round(scores[doc_idx], 6),
                    corpus[doc_idx],
                ])

    # Metrics
    all_scores = []
    for query in queries:
        all_scores.extend(bm25.score(query))

    metrics = {
        "n_documents": len(corpus),
        "n_queries": len(queries),
        "k1": k1,
        "b": b,
        "mean_score": float(sum(all_scores) / len(all_scores)) if all_scores else 0.0,
        "max_score": float(max(all_scores)) if all_scores else 0.0,
        "limitations": [
            "Synthetic corpus only",
            "Simple whitespace tokenizer",
            "No stemming or stop-word removal",
            "Single seed — no variance estimate",
        ],
    }
    return metrics
