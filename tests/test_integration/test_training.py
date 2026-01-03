"""Integration tests for actual training (requires FastText)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from babelvec import BabelVec
from babelvec.training import train_monolingual
from babelvec.training.config import TrainingConfig


@pytest.fixture
def small_corpus():
    """Create a small corpus for fast training."""
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a fast red fox leaps across the sleepy hound",
        "the dog chases the cat around the house",
        "cats and dogs are common household pets",
        "the weather is nice today",
        "it is a beautiful sunny day",
        "programming is fun and rewarding",
        "python is a popular programming language",
        "machine learning uses data to make predictions",
        "natural language processing analyzes text",
        "hello world this is a test",
        "testing the embedding model",
    ] * 50  # Repeat for minimum training data

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for sent in sentences:
            f.write(sent + "\n")
        return Path(f.name)


@pytest.fixture
def fast_config():
    """Fast training config for tests."""
    return TrainingConfig(
        dim=50,  # Small dimension
        epochs=1,  # Single epoch
        min_count=1,  # Include all words
        minn=2,
        maxn=4,
        threads=2,
        verbose=0,
    )


class TestActualTraining:
    """Tests that actually train FastText models."""

    @pytest.mark.slow
    def test_train_monolingual(self, small_corpus, fast_config):
        """Test training a monolingual model."""
        model = train_monolingual(
            lang="en",
            corpus_path=small_corpus,
            config=fast_config,
        )

        assert model.lang == "en"
        assert model.dim == 50
        assert model.vocab_size > 0

        # Test word vector
        vec = model.get_word_vector("dog")
        assert vec.shape == (50,)

        # Test sentence vector
        sent_vec = model.get_sentence_vector("the dog", method="average")
        assert sent_vec.shape == (50,)

    @pytest.mark.slow
    def test_save_and_load(self, small_corpus, fast_config, tmp_path):
        """Test saving and loading a model."""
        model = train_monolingual(
            lang="en",
            corpus_path=small_corpus,
            config=fast_config,
        )

        # Save with language in filename
        save_path = tmp_path / "en_model.bin"
        model.save(save_path)

        assert save_path.exists()

        # Load
        loaded = BabelVec.load(save_path)

        assert loaded.lang == "en"  # Inferred from filename
        assert loaded.dim == model.dim

        # Vectors should be the same
        vec1 = model.get_word_vector("dog")
        vec2 = loaded.get_word_vector("dog")
        assert np.allclose(vec1, vec2)

    @pytest.mark.slow
    def test_position_aware_sentence_vectors(self, small_corpus, fast_config):
        """Test that position-aware methods work with real model."""
        model = train_monolingual(
            lang="en",
            corpus_path=small_corpus,
            config=fast_config,
        )

        sent1 = "dog chases cat"
        sent2 = "cat chases dog"

        # Average should give similar results (same words)
        vec1_avg = model.get_sentence_vector(sent1, method="average")
        vec2_avg = model.get_sentence_vector(sent2, method="average")
        sim_avg = BabelVec.cosine_similarity(vec1_avg, vec2_avg)

        # RoPE should give different results
        vec1_rope = model.get_sentence_vector(sent1, method="rope")
        vec2_rope = model.get_sentence_vector(sent2, method="rope")
        sim_rope = BabelVec.cosine_similarity(vec1_rope, vec2_rope)

        # Average similarity should be very high (same words)
        assert sim_avg > 0.9

        # RoPE similarity should be lower (different order)
        assert sim_rope < sim_avg

    @pytest.mark.slow
    def test_most_similar(self, small_corpus, fast_config):
        """Test finding similar words."""
        model = train_monolingual(
            lang="en",
            corpus_path=small_corpus,
            config=fast_config,
        )

        similar = model.most_similar("dog", topn=5)

        assert len(similar) == 5
        assert all(isinstance(w, str) for w, _ in similar)
        assert all(isinstance(s, float) for _, s in similar)


# Mark all tests in this module as slow by default
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
