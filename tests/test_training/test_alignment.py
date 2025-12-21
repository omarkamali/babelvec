"""Tests for alignment methods."""

import numpy as np
import pytest
from unittest.mock import Mock

from babelvec.training.alignment.procrustes import ProcrustesAligner
from babelvec.training.alignment.infonce import InfoNCEAligner
from babelvec.training.alignment.ensemble import EnsembleAligner


def create_mock_model(lang, dim=100):
    """Create a mock BabelVec model."""
    mock = Mock()
    mock.lang = lang
    mock.dim = dim
    mock._ft = Mock()
    mock.metadata = {}
    mock.max_seq_len = 512

    # Create deterministic embeddings
    np.random.seed(hash(lang) % 2**32)
    base_matrix = np.random.randn(dim, dim).astype(np.float32)

    def get_sentence_vector(sent, method="average"):
        np.random.seed(hash(sent) % 2**32)
        vec = np.random.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    mock.get_sentence_vector = get_sentence_vector
    return mock


class TestProcrustesAligner:
    """Tests for Procrustes alignment."""

    def test_init(self):
        """Test aligner initialization."""
        aligner = ProcrustesAligner(reference_lang="en")
        assert aligner.reference_lang == "en"

    def test_compute_projections(self):
        """Test projection computation."""
        aligner = ProcrustesAligner(reference_lang="en")

        models = {
            "en": create_mock_model("en"),
            "fr": create_mock_model("fr"),
        }

        parallel_data = {
            ("en", "fr"): [
                ("hello", "bonjour"),
                ("world", "monde"),
                ("cat", "chat"),
                ("dog", "chien"),
            ] * 10  # Need enough data
        }

        projections = aligner.compute_projections(models, parallel_data)

        assert "en" in projections
        assert "fr" in projections
        assert projections["en"].shape == (100, 100)
        assert projections["fr"].shape == (100, 100)

        # Reference language should have identity
        assert np.allclose(projections["en"], np.eye(100))

    def test_projection_is_orthogonal(self):
        """Test that projection is orthogonal."""
        aligner = ProcrustesAligner()

        models = {
            "en": create_mock_model("en"),
            "fr": create_mock_model("fr"),
        }

        parallel_data = {
            ("en", "fr"): [("hello", "bonjour")] * 50
        }

        projections = aligner.compute_projections(models, parallel_data)

        # Check orthogonality: R @ R.T = I
        R = projections["fr"]
        assert np.allclose(R @ R.T, np.eye(100), atol=1e-5)


class TestInfoNCEAligner:
    """Tests for InfoNCE alignment."""

    def test_init(self):
        """Test aligner initialization."""
        aligner = InfoNCEAligner(epochs=3, batch_size=32)
        assert aligner.epochs == 3
        assert aligner.batch_size == 32

    def test_compute_projections(self):
        """Test projection computation."""
        aligner = InfoNCEAligner(epochs=2, batch_size=8)

        models = {
            "en": create_mock_model("en"),
            "fr": create_mock_model("fr"),
        }

        parallel_data = {
            ("en", "fr"): [("hello", "bonjour")] * 50
        }

        projections = aligner.compute_projections(models, parallel_data)

        assert "en" in projections
        assert "fr" in projections
        assert projections["fr"].shape == (100, 100)


class TestEnsembleAligner:
    """Tests for ensemble alignment."""

    def test_init(self):
        """Test aligner initialization."""
        aligner = EnsembleAligner(procrustes_weight=0.8, infonce_weight=0.2)
        assert aligner.procrustes_weight == 0.8
        assert aligner.infonce_weight == 0.2

    def test_init_invalid_weights_raises(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            EnsembleAligner(procrustes_weight=0.5, infonce_weight=0.3)

    def test_compute_projections(self):
        """Test ensemble projection computation."""
        aligner = EnsembleAligner(
            procrustes_weight=0.8,
            infonce_weight=0.2,
            infonce_epochs=1,
        )

        models = {
            "en": create_mock_model("en"),
            "fr": create_mock_model("fr"),
        }

        parallel_data = {
            ("en", "fr"): [("hello", "bonjour")] * 50
        }

        projections = aligner.compute_projections(models, parallel_data)

        assert "en" in projections
        assert "fr" in projections

        # Should be orthogonal
        R = projections["fr"]
        assert np.allclose(R @ R.T, np.eye(100), atol=1e-5)
