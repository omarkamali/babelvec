"""Tests for intrinsic evaluation metrics."""

import numpy as np
import pytest
from unittest.mock import Mock

from babelvec.evaluation.intrinsic import (
    word_similarity_eval,
    analogy_eval,
    oov_coverage,
    embedding_quality_metrics,
)


def create_mock_model(vocab_words, dim=100):
    """Create a mock model for testing."""
    mock = Mock()
    mock.dim = dim
    mock.words = vocab_words

    # Create deterministic vectors
    def get_word_vector(word):
        np.random.seed(hash(word) % 2**32)
        return np.random.randn(dim).astype(np.float32)

    def get_word_vectors(words):
        return np.array([get_word_vector(w) for w in words])

    mock.get_word_vector = get_word_vector
    mock.get_word_vectors = get_word_vectors
    mock.most_similar = lambda w, topn: [(f"similar_{i}", 0.9 - i * 0.1) for i in range(topn)]

    return mock


class TestWordSimilarityEval:
    """Tests for word similarity evaluation."""

    def test_basic_evaluation(self):
        """Test basic word similarity evaluation."""
        model = create_mock_model(["dog", "cat", "house"])

        pairs = [
            ("dog", "cat", 0.8),
            ("dog", "house", 0.3),
        ]

        results = word_similarity_eval(model, pairs)

        assert "spearman" in results
        assert "coverage" in results
        assert "n_pairs" in results
        assert results["n_pairs"] == 2
        assert results["coverage"] == 1.0

    def test_empty_pairs(self):
        """Test with empty pairs."""
        model = create_mock_model(["dog"])
        results = word_similarity_eval(model, [])

        assert results["spearman"] == 0.0
        assert results["n_pairs"] == 0


class TestAnalogyEval:
    """Tests for analogy evaluation."""

    def test_basic_evaluation(self):
        """Test basic analogy evaluation."""
        model = create_mock_model(["king", "queen", "man", "woman"])

        analogies = [
            ("king", "queen", "man", "woman"),
        ]

        results = analogy_eval(model, analogies)

        assert "accuracy" in results
        assert "correct" in results
        assert "total" in results


class TestOOVCoverage:
    """Tests for OOV coverage."""

    def test_full_coverage(self):
        """Test with full vocabulary coverage."""
        model = create_mock_model(["dog", "cat", "house"])

        results = oov_coverage(model, ["dog", "cat"])

        assert results["coverage"] == 1.0
        assert results["oov"] == 0

    def test_partial_coverage(self):
        """Test with partial coverage."""
        model = create_mock_model(["dog", "cat"])

        results = oov_coverage(model, ["dog", "cat", "unknown"])

        assert results["coverage"] == 2 / 3
        assert results["oov"] == 1
        assert results["in_vocab"] == 2


class TestEmbeddingQualityMetrics:
    """Tests for embedding quality metrics."""

    def test_basic_metrics(self):
        """Test basic quality metrics."""
        model = create_mock_model(["word1", "word2", "word3"])

        results = embedding_quality_metrics(model, n_samples=3)

        assert "mean_norm" in results
        assert "std_norm" in results
        assert "avg_pairwise_similarity" in results
        assert results["n_samples"] == 3
