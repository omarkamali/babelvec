"""Tests for sentence encoder."""

import numpy as np
import pytest

from babelvec.core.sentence_encoder import SentenceEncoder, MultiMethodEncoder


class TestSentenceEncoder:
    """Tests for SentenceEncoder."""

    def test_init(self):
        """Test encoder initialization."""
        encoder = SentenceEncoder(dim=100, method="rope")
        assert encoder.dim == 100
        assert encoder.method == "rope"

    def test_encode_mean_pooling(self):
        """Test mean pooling."""
        encoder = SentenceEncoder(dim=100, method="average", normalize=False)
        word_vecs = np.array([
            [1.0, 0.0, 0.0] + [0.0] * 97,
            [0.0, 1.0, 0.0] + [0.0] * 97,
            [0.0, 0.0, 1.0] + [0.0] * 97,
        ])

        result = encoder.encode(word_vecs, pooling="mean")
        expected = np.array([1/3, 1/3, 1/3] + [0.0] * 97)

        assert np.allclose(result, expected)

    def test_encode_max_pooling(self):
        """Test max pooling."""
        encoder = SentenceEncoder(dim=100, method="average", normalize=False)
        word_vecs = np.array([
            [1.0, 0.0, 0.5] + [0.0] * 97,
            [0.0, 2.0, 0.0] + [0.0] * 97,
            [0.5, 0.0, 1.0] + [0.0] * 97,
        ])

        result = encoder.encode(word_vecs, pooling="max")
        expected = np.array([1.0, 2.0, 1.0] + [0.0] * 97)

        assert np.allclose(result, expected)

    def test_encode_with_normalization(self):
        """Test that normalization produces unit vectors."""
        encoder = SentenceEncoder(dim=100, method="average", normalize=True)
        word_vecs = np.random.randn(5, 100)

        result = encoder.encode(word_vecs)
        norm = np.linalg.norm(result)

        assert np.isclose(norm, 1.0)

    def test_encode_empty_returns_zeros(self):
        """Test that empty input returns zero vector."""
        encoder = SentenceEncoder(dim=100, method="average")
        result = encoder.encode(np.array([]).reshape(0, 100))

        assert result.shape == (100,)
        assert np.allclose(result, 0)

    def test_position_aware_encoding(self):
        """Test that position-aware methods give different results for reordered input."""
        encoder_rope = SentenceEncoder(dim=100, method="rope", normalize=False)
        encoder_avg = SentenceEncoder(dim=100, method="average", normalize=False)

        # Create word vectors
        word_vecs = np.random.randn(3, 100)
        word_vecs_reversed = word_vecs[::-1]

        # Average should give same result regardless of order
        avg_result = encoder_avg.encode(word_vecs)
        avg_reversed = encoder_avg.encode(word_vecs_reversed)
        assert np.allclose(avg_result, avg_reversed)

        # RoPE should give different results
        rope_result = encoder_rope.encode(word_vecs)
        rope_reversed = encoder_rope.encode(word_vecs_reversed)
        assert not np.allclose(rope_result, rope_reversed)

    def test_similarity(self):
        """Test similarity computation."""
        encoder = SentenceEncoder(dim=100, method="average")

        vec1 = np.array([1.0, 0.0, 0.0] + [0.0] * 97)
        vec2 = np.array([1.0, 0.0, 0.0] + [0.0] * 97)
        vec3 = np.array([0.0, 1.0, 0.0] + [0.0] * 97)

        assert np.isclose(encoder.similarity(vec1, vec2), 1.0)
        assert np.isclose(encoder.similarity(vec1, vec3), 0.0)


class TestMultiMethodEncoder:
    """Tests for MultiMethodEncoder."""

    def test_init(self):
        """Test initialization."""
        encoder = MultiMethodEncoder(dim=100)
        assert "average" in encoder.methods
        assert "rope" in encoder.methods
        assert "sinusoidal" in encoder.methods
        assert "decay" in encoder.methods

    def test_encode_all_methods(self):
        """Test encoding with all methods."""
        encoder = MultiMethodEncoder(dim=100)
        word_vecs = np.random.randn(5, 100)

        results = encoder.encode_all_methods(word_vecs)

        assert len(results) == 4
        for method, vec in results.items():
            assert vec.shape == (100,)

    def test_compare_methods(self):
        """Test method comparison."""
        encoder = MultiMethodEncoder(dim=100)
        word_vecs1 = np.random.randn(5, 100)
        word_vecs2 = np.random.randn(5, 100)

        results = encoder.compare_methods(word_vecs1, word_vecs2)

        assert len(results) == 4
        for method, sim in results.items():
            assert -1 <= sim <= 1
