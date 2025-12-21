#!/usr/bin/env python
"""
Basic BabelVec Usage Example

This example demonstrates:
1. Training a simple model
2. Getting word and sentence vectors
3. Position-aware sentence encoding
4. Finding similar words
"""

import tempfile
from pathlib import Path

from babelvec import BabelVec
from babelvec.training import train_monolingual
from babelvec.training.config import TrainingConfig


def create_sample_corpus():
    """Create a sample corpus for demonstration."""
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
        "deep learning models require lots of data",
        "neural networks can learn complex patterns",
    ] * 100  # Repeat for sufficient training data

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for sent in sentences:
            f.write(sent + "\n")
        return Path(f.name)


def main():
    print("=" * 60)
    print("BabelVec Basic Usage Example")
    print("=" * 60)

    # Create sample corpus
    print("\n1. Creating sample corpus...")
    corpus_path = create_sample_corpus()
    print(f"   Corpus created at: {corpus_path}")

    # Configure training (fast settings for demo)
    config = TrainingConfig(
        dim=100,  # Small dimension for speed
        epochs=3,
        min_count=1,
        verbose=0,
    )

    # Train model
    print("\n2. Training model...")
    model = train_monolingual(
        lang="en",
        corpus_path=corpus_path,
        config=config,
    )
    print(f"   Model trained: {model}")
    print(f"   Vocabulary size: {model.vocab_size}")

    # Get word vectors
    print("\n3. Word Vectors:")
    for word in ["dog", "cat", "programming"]:
        vec = model.get_word_vector(word)
        print(f"   '{word}': shape={vec.shape}, norm={vec.dot(vec)**0.5:.3f}")

    # Find similar words
    print("\n4. Similar Words:")
    for word in ["dog", "programming"]:
        similar = model.most_similar(word, topn=5)
        print(f"   Similar to '{word}':")
        for w, score in similar:
            print(f"      {w}: {score:.3f}")

    # Sentence vectors with different methods
    print("\n5. Sentence Vectors (Position-Aware):")
    sent1 = "the dog bites the man"
    sent2 = "the man bites the dog"

    print(f"   Sentence 1: '{sent1}'")
    print(f"   Sentence 2: '{sent2}'")
    print()

    for method in ["average", "rope", "sinusoidal", "decay"]:
        vec1 = model.get_sentence_vector(sent1, method=method)
        vec2 = model.get_sentence_vector(sent2, method=method)
        sim = BabelVec.cosine_similarity(vec1, vec2)
        print(f"   {method:12s}: similarity = {sim:.4f}")

    print()
    print("   Note: 'average' gives high similarity (same words)")
    print("   Position-aware methods (rope, sinusoidal, decay) give lower")
    print("   similarity because word ORDER matters!")

    # Cleanup
    corpus_path.unlink()
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
