"""Tests for positional encoding implementations."""

import numpy as np
import pytest

from babelvec.core.positional_encoding import (
    RoPEEncoding,
    SinusoidalEncoding,
    DecayEncoding,
    get_positional_encoding,
)


class TestRoPEEncoding:
    """Tests for RoPE positional encoding."""

    def test_init(self):
        """Test RoPE initialization."""
        rope = RoPEEncoding(dim=100)
        assert rope.dim == 100
        assert rope.max_seq_len == 512

    def test_init_odd_dim_raises(self):
        """Test that odd dimension raises error."""
        with pytest.raises(ValueError, match="even dimension"):
            RoPEEncoding(dim=101)

    def test_encode_single_vector(self):
        """Test encoding a single vector."""
        rope = RoPEEncoding(dim=100)
        vec = np.random.randn(100)
        encoded = rope.encode(vec)

        assert encoded.shape == (100,)
        # RoPE at position 0 should still modify the vector
        # (unless the vector happens to align with rotation)

    def test_encode_sequence(self):
        """Test encoding a sequence of vectors."""
        rope = RoPEEncoding(dim=100)
        seq = np.random.randn(5, 100)
        encoded = rope.encode(seq)

        assert encoded.shape == (5, 100)

    def test_position_sensitivity(self):
        """Test that different positions give different encodings."""
        rope = RoPEEncoding(dim=100)
        vec = np.random.randn(100)

        # Same vector at different positions
        encoded_0 = rope.encode(vec[np.newaxis, :], positions=np.array([0]))[0]
        encoded_5 = rope.encode(vec[np.newaxis, :], positions=np.array([5]))[0]

        # Should be different
        assert not np.allclose(encoded_0, encoded_5)

    def test_preserves_norm_approximately(self):
        """Test that RoPE approximately preserves vector norm."""
        rope = RoPEEncoding(dim=100)
        vec = np.random.randn(100)
        encoded = rope.encode(vec)

        # Norms should be similar (rotation preserves norm)
        assert np.isclose(np.linalg.norm(vec), np.linalg.norm(encoded), rtol=0.01)


class TestSinusoidalEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_init(self):
        """Test sinusoidal initialization."""
        sin_enc = SinusoidalEncoding(dim=100)
        assert sin_enc.dim == 100

    def test_encode_single_vector(self):
        """Test encoding a single vector."""
        sin_enc = SinusoidalEncoding(dim=100)
        vec = np.random.randn(100)
        encoded = sin_enc.encode(vec)

        assert encoded.shape == (100,)
        # Should be different (positional signal added)
        assert not np.allclose(encoded, vec)

    def test_encode_sequence(self):
        """Test encoding a sequence."""
        sin_enc = SinusoidalEncoding(dim=100)
        seq = np.random.randn(5, 100)
        encoded = sin_enc.encode(seq)

        assert encoded.shape == (5, 100)

    def test_position_sensitivity(self):
        """Test position sensitivity."""
        sin_enc = SinusoidalEncoding(dim=100)
        vec = np.random.randn(100)

        encoded_0 = sin_enc.encode(vec[np.newaxis, :], positions=np.array([0]))[0]
        encoded_5 = sin_enc.encode(vec[np.newaxis, :], positions=np.array([5]))[0]

        assert not np.allclose(encoded_0, encoded_5)


class TestDecayEncoding:
    """Tests for decay positional encoding."""

    def test_init(self):
        """Test decay initialization."""
        decay = DecayEncoding(dim=100, decay_rate=0.1)
        assert decay.dim == 100
        assert decay.decay_rate == 0.1

    def test_encode_single_vector(self):
        """Test encoding a single vector."""
        decay = DecayEncoding(dim=100)
        vec = np.random.randn(100)
        encoded = decay.encode(vec)

        assert encoded.shape == (100,)

    def test_decay_effect(self):
        """Test that later positions have smaller weights."""
        decay = DecayEncoding(dim=100, decay_rate=0.5)
        seq = np.ones((5, 100))
        encoded = decay.encode(seq)

        # Later positions should have smaller norms
        norms = np.linalg.norm(encoded, axis=1)
        assert norms[0] > norms[1] > norms[2] > norms[3] > norms[4]


class TestGetPositionalEncoding:
    """Tests for the factory function."""

    def test_get_rope(self):
        """Test getting RoPE encoder."""
        enc = get_positional_encoding("rope", dim=100)
        assert isinstance(enc, RoPEEncoding)

    def test_get_sinusoidal(self):
        """Test getting sinusoidal encoder."""
        enc = get_positional_encoding("sinusoidal", dim=100)
        assert isinstance(enc, SinusoidalEncoding)

    def test_get_decay(self):
        """Test getting decay encoder."""
        enc = get_positional_encoding("decay", dim=100)
        assert isinstance(enc, DecayEncoding)

    def test_get_average_returns_none(self):
        """Test that 'average' returns None."""
        enc = get_positional_encoding("average", dim=100)
        assert enc is None

    def test_unknown_method_raises(self):
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown"):
            get_positional_encoding("unknown", dim=100)
