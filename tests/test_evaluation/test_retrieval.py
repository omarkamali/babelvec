"""Tests for cross-lingual retrieval evaluation."""

import numpy as np
import pytest
from unittest.mock import Mock

from babelvec.evaluation.retrieval_eval import (
    cross_lingual_retrieval,
    retrieval_accuracy,
    alignment_quality_score,
)


def create_mock_model(lang, dim=100, aligned=False):
    """Create a mock model for testing."""
    mock = Mock()
    mock.lang = lang
    mock.dim = dim

    # If aligned, use similar vectors for parallel sentences
    # Otherwise, use random vectors
    def get_sentence_vector(sent, method="average"):
        if aligned:
            # Use sentence hash but add language-specific offset
            np.random.seed(hash(sent) % 2**32)
            vec = np.random.randn(dim).astype(np.float32)
        else:
            # Completely random
            np.random.seed(hash(f"{lang}_{sent}") % 2**32)
            vec = np.random.randn(dim).astype(np.float32)

        return vec / (np.linalg.norm(vec) + 1e-8)

    mock.get_sentence_vector = get_sentence_vector
    return mock


class TestCrossLingualRetrieval:
    """Tests for cross-lingual retrieval."""

    def test_basic_retrieval(self):
        """Test basic retrieval evaluation."""
        model_en = create_mock_model("en", aligned=True)
        model_fr = create_mock_model("fr", aligned=True)

        parallel = [
            ("hello", "bonjour"),
            ("world", "monde"),
            ("cat", "chat"),
        ]

        results = cross_lingual_retrieval(model_en, model_fr, parallel)

        assert "recall@1" in results
        assert "recall@5" in results
        assert "mrr" in results
        assert "avg_parallel_similarity" in results
        assert results["n_pairs"] == 3

    def test_empty_parallel_data(self):
        """Test with empty parallel data."""
        model_en = create_mock_model("en")
        model_fr = create_mock_model("fr")

        results = cross_lingual_retrieval(model_en, model_fr, [])

        assert results["recall@1"] == 0.0


class TestRetrievalAccuracy:
    """Tests for retrieval accuracy across language pairs."""

    def test_multiple_pairs(self):
        """Test retrieval for multiple language pairs."""
        models = {
            "en": create_mock_model("en"),
            "fr": create_mock_model("fr"),
        }

        parallel_data = {
            ("en", "fr"): [
                ("hello", "bonjour"),
                ("world", "monde"),
            ]
        }

        results = retrieval_accuracy(models, parallel_data)

        assert "en->fr" in results
        assert "fr->en" in results


class TestAlignmentQualityScore:
    """Tests for alignment quality score."""

    def test_quality_score_range(self):
        """Test that quality score is in valid range."""
        model_en = create_mock_model("en")
        model_fr = create_mock_model("fr")

        parallel = [("hello", "bonjour")] * 10

        score = alignment_quality_score(model_en, model_fr, parallel)

        assert 0 <= score <= 1
