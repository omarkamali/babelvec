
import pytest
import os
from unittest.mock import patch, MagicMock
from babelvec.training.config import TrainingConfig
from babelvec.training.monolingual import train_monolingual, train_multiple_languages
from babelvec.training.multilingual import train_multilingual
from pathlib import Path

@pytest.fixture
def dummy_corpus(tmp_path):
    corpus_path = tmp_path / "dummy_corpus.txt"
    corpus_path.write_text("hello world\nthis is a test corpus\n")
    return corpus_path

def test_monolingual_threads_propagation(dummy_corpus):
    """Test that thread count is correctly passed to FastText."""
    with patch('fasttext.train_unsupervised') as mock_train:
        # 1. Test explicit threads parameter
        train_monolingual("en", dummy_corpus, threads=2, verbose=0)
        _, kwargs = mock_train.call_args
        assert kwargs["thread"] == 2
        
        mock_train.reset_mock()
        
        # 2. Test thread alias in kwargs
        train_monolingual("en", dummy_corpus, thread=3, verbose=0)
        _, kwargs = mock_train.call_args
        assert kwargs["thread"] == 3
        
        mock_train.reset_mock()
        
        # 3. Test threads in config
        config = TrainingConfig(threads=4)
        train_monolingual("en", dummy_corpus, config=config, verbose=0)
        _, kwargs = mock_train.call_args
        assert kwargs["thread"] == 4

def test_multiple_languages_threads_splitting(tmp_path):
    """Test that threads are correctly divided among parallel jobs."""
    en_corpus = tmp_path / "en.txt"
    fr_corpus = tmp_path / "fr.txt"
    en_corpus.write_text("hello\n")
    fr_corpus.write_text("bonjour\n")
    
    languages = {"en": en_corpus, "fr": fr_corpus}
    
    with patch('fasttext.train_unsupervised') as mock_train:
        # Total 4 threads, 2 workers -> 2 threads per job
        config = TrainingConfig(threads=4)
        train_multiple_languages(languages, config=config, parallel=True, max_workers=2)
        
        assert mock_train.call_count == 2
        for call in mock_train.call_args_list:
            _, kwargs = call
            assert kwargs["thread"] == 2

def test_multilingual_thread_alias(dummy_corpus):
    """Test that the 'thread' alias works in train_multilingual."""
    languages = ["en"]
    corpus_paths = {"en": dummy_corpus}
    
    with patch('fasttext.train_unsupervised') as mock_train:
        train_multilingual(languages, corpus_paths, thread=5, alignment="none", verbose=0)
        _, kwargs = mock_train.call_args
        assert kwargs["thread"] == 5

def test_config_preservation_in_parallel(tmp_path):
    """Test that all config parameters are preserved when splitting for parallel training."""
    en_corpus = tmp_path / "en.txt"
    fr_corpus = tmp_path / "fr.txt"
    en_corpus.write_text("hello\n")
    fr_corpus.write_text("bonjour\n")
    
    languages = {"en": en_corpus, "fr": fr_corpus}
    
    # Custom config with many non-default values
    config = TrainingConfig(
        threads=8,
        dim=128,
        epochs=7,
        ws=3,
        neg=15,
        model_type="cbow",
        loss="hs"
    )
    
    with patch('fasttext.train_unsupervised') as mock_train:
        train_multiple_languages(languages, config=config, parallel=True, max_workers=2)
        
        assert mock_train.call_count == 2
        for call in mock_train.call_args_list:
            _, kwargs = call
            # Check threads were split
            assert kwargs["thread"] == 4
            # Check other params were preserved
            assert kwargs["dim"] == 128
            assert kwargs["epoch"] == 7
            assert kwargs["ws"] == 3
            assert kwargs["neg"] == 15
            assert kwargs["model"] == "cbow"
            assert kwargs["loss"] == "hs"
