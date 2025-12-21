"""Tests for BabelVec model."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from babelvec.core.model import BabelVec


class TestBabelVecBasic:
    """Basic tests for BabelVec without FastText."""

    def test_init(self):
        """Test model initialization."""
        model = BabelVec(lang="en", dim=100)
        assert model.lang == "en"
        assert model.dim == 100
        assert model.vocab_size == 0
        assert not model.is_aligned

    def test_repr(self):
        """Test string representation."""
        model = BabelVec(lang="en", dim=100)
        repr_str = repr(model)
        assert "en" in repr_str
        assert "100" in repr_str

    def test_tokenize(self):
        """Test tokenization."""
        model = BabelVec(lang="en", dim=100)
        tokens = model.tokenize("hello world test")
        assert tokens == ["hello", "world", "test"]

    def test_cosine_similarity_static(self):
        """Test static cosine similarity."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])

        assert np.isclose(BabelVec.cosine_similarity(vec1, vec2), 1.0)
        assert np.isclose(BabelVec.cosine_similarity(vec1, vec3), 0.0)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec_zero = np.array([0.0, 0.0, 0.0])

        assert BabelVec.cosine_similarity(vec1, vec_zero) == 0.0

    def test_set_projection(self):
        """Test setting projection matrix."""
        model = BabelVec(lang="en", dim=100)  # Must be even for RoPE
        proj = np.eye(100)

        model.set_projection(proj)
        assert model.is_aligned
        assert model._projection is not None

    def test_set_projection_wrong_shape_raises(self):
        """Test that wrong projection shape raises error."""
        model = BabelVec(lang="en", dim=100)
        proj = np.eye(50)  # Wrong shape

        with pytest.raises(ValueError):
            model.set_projection(proj)

    def test_clear_projection(self):
        """Test clearing projection."""
        model = BabelVec(lang="en", dim=100)
        model.set_projection(np.eye(100))
        assert model.is_aligned

        model.clear_projection()
        assert not model.is_aligned


class TestBabelVecWithMockFastText:
    """Tests for BabelVec with mocked FastText."""

    @pytest.fixture
    def mock_ft(self):
        """Create a mock FastText wrapper."""
        mock = Mock()
        mock.dim = 100
        mock.vocab_size = 1000
        mock.words = ["hello", "world", "test"]

        def get_word_vector(word):
            # Return deterministic vectors based on word
            np.random.seed(hash(word) % 2**32)
            return np.random.randn(100).astype(np.float32)

        mock.get_word_vector = get_word_vector
        mock.get_word_vectors = lambda words: np.array([get_word_vector(w) for w in words])
        mock.get_nearest_neighbors = lambda w, k: [(0.9, "similar1"), (0.8, "similar2")]
        mock.__contains__ = lambda self, word: word in ["hello", "world", "test"]

        return mock

    def test_with_fasttext(self, mock_ft):
        """Test model with FastText."""
        model = BabelVec(fasttext_model=mock_ft, lang="en")

        assert model.dim == 100
        assert model.vocab_size == 1000

    def test_get_word_vector(self, mock_ft):
        """Test getting word vector."""
        model = BabelVec(fasttext_model=mock_ft, lang="en")
        vec = model.get_word_vector("hello")

        assert vec.shape == (100,)

    def test_get_sentence_vector(self, mock_ft):
        """Test getting sentence vector."""
        model = BabelVec(fasttext_model=mock_ft, lang="en", dim=100)
        vec = model.get_sentence_vector("hello world", method="average")

        assert vec.shape == (100,)
        # Should be normalized
        assert np.isclose(np.linalg.norm(vec), 1.0)

    def test_get_sentence_vector_rope(self, mock_ft):
        """Test sentence vector with RoPE."""
        model = BabelVec(fasttext_model=mock_ft, lang="en", dim=100)

        vec1 = model.get_sentence_vector("hello world", method="rope")
        vec2 = model.get_sentence_vector("world hello", method="rope")

        # Should be different due to position encoding
        assert not np.allclose(vec1, vec2)

    def test_similarity(self, mock_ft):
        """Test text similarity."""
        model = BabelVec(fasttext_model=mock_ft, lang="en", dim=100)

        sim = model.similarity("hello world", "hello world", method="average")
        assert np.isclose(sim, 1.0)

    def test_most_similar(self, mock_ft):
        """Test most similar words."""
        model = BabelVec(fasttext_model=mock_ft, lang="en")
        similar = model.most_similar("hello", topn=2)

        assert len(similar) == 2
        assert similar[0][0] == "similar1"

    def test_contains(self, mock_ft):
        """Test word containment check."""
        model = BabelVec(fasttext_model=mock_ft, lang="en")

        # The mock's words list contains these
        assert "hello" in model
        assert "nonexistent" not in model
