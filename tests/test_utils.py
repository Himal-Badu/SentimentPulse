"""
Unit tests for SentimentPulse utility functions
Built by Himal Badu, AI Founder
"""

import pytest
from sentimentpulse.utils import (
    validate_text_input,
    calculate_sentiment_distribution,
    truncate_text,
    generate_cache_key,
)


class TestValidateTextInput:
    """Tests for text input validation."""
    
    def test_valid_text(self):
        """Test valid text input passes validation."""
        is_valid, error = validate_text_input("This is a valid text!")
        assert is_valid is True
        assert error == ""
    
    def test_empty_text(self):
        """Test empty text fails validation."""
        is_valid, error = validate_text_input("")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_whitespace_only(self):
        """Test whitespace-only text fails validation."""
        is_valid, error = validate_text_input("   ")
        assert is_valid is False
        assert "whitespace" in error.lower()
    
    def test_max_length_exceeded(self):
        """Test text exceeding max length fails validation."""
        long_text = "a" * 10001
        is_valid, error = validate_text_input(long_text, max_length=10000)
        assert is_valid is False
        assert "exceeds" in error.lower()
    
    def test_text_at_max_length(self):
        """Test text at exactly max length passes validation."""
        text = "a" * 10000
        is_valid, error = validate_text_input(text, max_length=10000)
        assert is_valid is True


class TestCalculateSentimentDistribution:
    """Tests for sentiment distribution calculation."""
    
    def test_empty_results(self):
        """Test empty results returns zeros."""
        result = calculate_sentiment_distribution([])
        assert result["total"] == 0
        assert result["positive"] == 0
        assert result["negative"] == 0
        assert result["neutral"] == 0
    
    def test_all_positive(self):
        """Test all positive sentiments."""
        results = [
            {"sentiment": "positive"},
            {"sentiment": "positive"},
            {"sentiment": "positive"},
        ]
        result = calculate_sentiment_distribution(results)
        assert result["total"] == 3
        assert result["positive"] == 3
        assert result["positive_pct"] == 100.0
    
    def test_mixed_sentiments(self):
        """Test mixed sentiment distribution."""
        results = [
            {"sentiment": "positive"},
            {"sentiment": "negative"},
            {"sentiment": "neutral"},
            {"sentiment": "positive"},
        ]
        result = calculate_sentiment_distribution(results)
        assert result["total"] == 4
        assert result["positive"] == 2
        assert result["negative"] == 1
        assert result["neutral"] == 1
        assert result["positive_pct"] == 50.0
    
    def test_percentages_sum_to_100(self):
        """Test percentages sum to 100."""
        results = [
            {"sentiment": "positive"},
            {"sentiment": "negative"},
            {"sentiment": "neutral"},
            {"sentiment": "neutral"},
        ]
        result = calculate_sentiment_distribution(results)
        total_pct = (
            result["positive_pct"] + 
            result["negative_pct"] + 
            result["neutral_pct"]
        )
        assert total_pct == 100.0


class TestTruncateText:
    """Tests for text truncation."""
    
    def test_short_text(self):
        """Test short text not truncated."""
        text = "Hello world"
        result = truncate_text(text, max_length=20)
        assert result == text
    
    def test_long_text_truncated(self):
        """Test long text is truncated."""
        text = "a" * 100
        result = truncate_text(text, max_length=20)
        assert len(result) == 20
        assert result.endswith("...")
    
    def test_custom_suffix(self):
        """Test custom truncation suffix."""
        text = "a" * 100
        result = truncate_text(text, max_length=20, suffix="…")
        assert len(result) == 20
        assert result.endswith("…")
    
    def test_exact_length(self):
        """Test text at exactly max length."""
        text = "Hello"
        result = truncate_text(text, max_length=5)
        assert result == text


class TestGenerateCacheKey:
    """Tests for cache key generation."""
    
    def test_same_text_same_key(self):
        """Test same text produces same cache key."""
        key1 = generate_cache_key("Hello world")
        key2 = generate_cache_key("Hello world")
        assert key1 == key2
    
    def test_different_text_different_key(self):
        """Test different texts produce different keys."""
        key1 = generate_cache_key("Hello world")
        key2 = generate_cache_key("Goodbye world")
        assert key1 != key2
    
    def test_prefix_applied(self):
        """Test custom prefix is applied."""
        key = generate_cache_key("test", prefix="custom")
        assert key.startswith("custom:")
    
    def test_default_prefix(self):
        """Test default prefix is 'sentiment'."""
        key = generate_cache_key("test")
        assert key.startswith("sentiment:")
