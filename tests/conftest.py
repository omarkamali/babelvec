"""Pytest fixtures for BabelVec tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_corpus():
    """Create a sample corpus file for testing."""
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
    ] * 100  # Repeat for sufficient training data

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for sent in sentences:
            f.write(sent + "\n")
        return Path(f.name)


@pytest.fixture
def sample_corpus_fr():
    """Create a sample French corpus for testing."""
    sentences = [
        "le renard brun rapide saute par-dessus le chien paresseux",
        "un renard rouge rapide bondit sur le chien endormi",
        "le chien poursuit le chat autour de la maison",
        "les chats et les chiens sont des animaux domestiques",
        "le temps est beau aujourd'hui",
        "c'est une belle journée ensoleillée",
        "la programmation est amusante et gratifiante",
        "python est un langage de programmation populaire",
    ] * 100

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for sent in sentences:
            f.write(sent + "\n")
        return Path(f.name)


@pytest.fixture
def parallel_data():
    """Create sample parallel data for alignment testing."""
    return {
        ("en", "fr"): [
            ("the dog", "le chien"),
            ("the cat", "le chat"),
            ("the house", "la maison"),
            ("good morning", "bonjour"),
            ("thank you", "merci"),
            ("hello world", "bonjour monde"),
            ("the weather is nice", "le temps est beau"),
            ("programming is fun", "la programmation est amusante"),
        ]
    }


@pytest.fixture
def word_similarity_pairs():
    """Sample word similarity pairs for evaluation."""
    return [
        ("dog", "cat", 0.8),
        ("dog", "house", 0.3),
        ("cat", "house", 0.2),
        ("quick", "fast", 0.9),
        ("nice", "beautiful", 0.85),
    ]


@pytest.fixture
def sentence_pairs_same_words():
    """Sentence pairs with same words but different order."""
    return [
        ("the dog bites the man", "the man bites the dog"),
        ("cat chases dog", "dog chases cat"),
        ("fox jumps over dog", "dog jumps over fox"),
    ]


@pytest.fixture
def random_embeddings():
    """Generate random embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 100).astype(np.float32)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
