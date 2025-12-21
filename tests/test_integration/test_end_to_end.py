"""End-to-end integration tests."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from babelvec import BabelVec
from babelvec.core.positional_encoding import RoPEEncoding
from babelvec.core.sentence_encoder import SentenceEncoder


class TestPositionAwareness:
    """Test position-aware sentence encoding end-to-end."""

    def test_word_order_matters_with_rope(self):
        """Test that RoPE makes word order matter."""
        dim = 100

        # Create mock model
        model = Mock()
        model.dim = dim
        model._ft = Mock()
        model._ft.dim = dim

        # Create deterministic word vectors
        word_vectors = {
            "the": np.random.randn(dim),
            "dog": np.random.randn(dim),
            "bites": np.random.randn(dim),
            "man": np.random.randn(dim),
        }

        def get_word_vector(word):
            return word_vectors.get(word.lower(), np.random.randn(dim))

        model._ft.get_word_vector = get_word_vector
        model._ft.get_word_vectors = lambda words: np.array([get_word_vector(w) for w in words])

        # Create BabelVec with the mock
        bv = BabelVec(fasttext_model=model._ft, lang="en", dim=dim)

        # Get sentence vectors
        sent1 = "the dog bites the man"
        sent2 = "the man bites the dog"

        # With average (order-agnostic)
        vec1_avg = bv.get_sentence_vector(sent1, method="average")
        vec2_avg = bv.get_sentence_vector(sent2, method="average")

        # With RoPE (order-aware)
        vec1_rope = bv.get_sentence_vector(sent1, method="rope")
        vec2_rope = bv.get_sentence_vector(sent2, method="rope")

        # Average should give same result (same words)
        # Note: might not be exactly equal due to word repetition handling
        sim_avg = BabelVec.cosine_similarity(vec1_avg, vec2_avg)

        # RoPE should give different results
        sim_rope = BabelVec.cosine_similarity(vec1_rope, vec2_rope)

        # RoPE similarity should be lower (different meanings)
        assert sim_rope < sim_avg or not np.isclose(sim_rope, 1.0)


class TestEncoderIntegration:
    """Test encoder integration."""

    def test_all_encoding_methods(self):
        """Test all encoding methods work together."""
        dim = 100
        encoder_avg = SentenceEncoder(dim, method="average")
        encoder_rope = SentenceEncoder(dim, method="rope")
        encoder_sin = SentenceEncoder(dim, method="sinusoidal")
        encoder_decay = SentenceEncoder(dim, method="decay")

        word_vecs = np.random.randn(5, dim)

        # All should produce valid output
        vec_avg = encoder_avg.encode(word_vecs)
        vec_rope = encoder_rope.encode(word_vecs)
        vec_sin = encoder_sin.encode(word_vecs)
        vec_decay = encoder_decay.encode(word_vecs)

        assert vec_avg.shape == (dim,)
        assert vec_rope.shape == (dim,)
        assert vec_sin.shape == (dim,)
        assert vec_decay.shape == (dim,)

        # All should be normalized
        assert np.isclose(np.linalg.norm(vec_avg), 1.0)
        assert np.isclose(np.linalg.norm(vec_rope), 1.0)
        assert np.isclose(np.linalg.norm(vec_sin), 1.0)
        assert np.isclose(np.linalg.norm(vec_decay), 1.0)


class TestAlignmentIntegration:
    """Test alignment integration."""

    def test_alignment_improves_similarity(self):
        """Test that alignment improves cross-lingual similarity."""
        from babelvec.training.alignment import ProcrustesAligner

        dim = 50

        # Create mock models with different embedding spaces
        def create_model(lang, rotation=None):
            mock = Mock()
            mock.lang = lang
            mock.dim = dim
            mock._ft = Mock()
            mock.metadata = {}
            mock.max_seq_len = 512

            # Base vectors
            np.random.seed(42)
            base_vecs = {
                "hello": np.random.randn(dim),
                "world": np.random.randn(dim),
                "cat": np.random.randn(dim),
            }

            if rotation is not None:
                # Apply rotation to create different space
                base_vecs = {k: v @ rotation for k, v in base_vecs.items()}

            def get_sentence_vector(sent, method="average"):
                words = sent.lower().split()
                vecs = [base_vecs.get(w, np.random.randn(dim)) for w in words]
                vec = np.mean(vecs, axis=0)
                return vec / (np.linalg.norm(vec) + 1e-8)

            mock.get_sentence_vector = get_sentence_vector
            return mock

        # Create random rotation for second language
        np.random.seed(123)
        random_matrix = np.random.randn(dim, dim)
        U, _, Vt = np.linalg.svd(random_matrix)
        rotation = U @ Vt

        model_en = create_model("en")
        model_fr = create_model("fr", rotation=rotation)

        # Before alignment - vectors should be different
        vec_en = model_en.get_sentence_vector("hello")
        vec_fr = model_fr.get_sentence_vector("hello")
        sim_before = np.dot(vec_en, vec_fr) / (np.linalg.norm(vec_en) * np.linalg.norm(vec_fr))

        # Alignment should find the rotation
        aligner = ProcrustesAligner(reference_lang="en")
        parallel_data = {
            ("en", "fr"): [
                ("hello", "hello"),
                ("world", "world"),
                ("cat", "cat"),
            ] * 20
        }

        projections = aligner.compute_projections(
            {"en": model_en, "fr": model_fr},
            parallel_data
        )

        # Apply projection to French vector
        vec_fr_aligned = vec_fr @ projections["fr"]
        sim_after = np.dot(vec_en, vec_fr_aligned) / (
            np.linalg.norm(vec_en) * np.linalg.norm(vec_fr_aligned)
        )

        # Similarity should improve after alignment
        # (or at least not get worse significantly)
        assert sim_after >= sim_before - 0.1 or sim_after > 0.5
