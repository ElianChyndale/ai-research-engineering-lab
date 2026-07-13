"""Tests for airelab.foundations.bm25."""

from __future__ import annotations

import pytest

from airelab.foundations.bm25 import BM25


@pytest.mark.unit
class TestBM25:
    def test_hand_computed_tiny_corpus(self) -> None:
        """Known BM25 scores for a tiny corpus."""
        corpus = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "cats and dogs are friends",
        ]
        bm25 = BM25(corpus)
        scores = bm25.score("cat")
        assert scores[0] > scores[1]  # "cat" exact match in doc 0
        assert scores[0] > scores[2]  # "cat" not in doc 2 ("cats" ≠ "cat")

    def test_exact_term_identifier(self) -> None:
        corpus = ["alpha beta", "gamma delta", "alpha gamma"]
        bm25 = BM25(corpus)
        scores = bm25.score("alpha")
        assert scores[0] > 0
        assert scores[1] == 0.0
        assert scores[2] > 0

    def test_repeated_term_saturation(self) -> None:
        """BM25 should saturate — repeating a term shouldn't increase score linearly."""
        doc1 = "cat " * 10
        doc2 = "cat " * 20
        bm25 = BM25([doc1, doc2])
        scores = bm25.score("cat")
        # doc2 should score higher but not 2x
        assert scores[1] > scores[0]
        assert scores[1] < scores[0] * 2

    def test_document_length_effect(self) -> None:
        """Longer documents with same term count should score lower."""
        bm25 = BM25(["cat", "cat and many other words here today"])
        scores = bm25.score("cat")
        assert scores[0] > scores[1]

    def test_deterministic_tie_breaking(self) -> None:
        """Same scores should be broken by document ID."""
        corpus = ["a", "a"]
        bm25 = BM25(corpus)
        scores = bm25.score("a")
        assert scores[0] == scores[1]

    def test_empty_query(self) -> None:
        bm25 = BM25(["hello world"])
        scores = bm25.score("")
        assert scores[0] == 0.0

    def test_no_matching_term(self) -> None:
        bm25 = BM25(["hello world"])
        scores = bm25.score("xyz")
        assert scores[0] == 0.0

    def test_rank_returns_indices(self) -> None:
        corpus = ["the cat", "the dog", "cats dogs"]
        bm25 = BM25(corpus)
        ranked = bm25.rank("cat")
        assert ranked[0] == 0  # "the cat" should rank first
        assert len(ranked) == 3

    def test_deterministic(self) -> None:
        corpus = ["hello world", "world hello", "hello hello"]
        bm25_1 = BM25(corpus)
        bm25_2 = BM25(corpus)
        assert bm25_1.score("hello") == bm25_2.score("hello")
