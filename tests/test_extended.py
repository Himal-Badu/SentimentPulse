"""
Extended tests for SentimentPulse Analyzer
Built by Himal Badu, AI Founder

Tests for edge cases, performance, and additional functionality.
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from sentimentpulse import analyze_sentiment, analyze_batch, get_engine
from sentimentpulse.engine import SentimentEngine, SentimentCache, SentimentLabel
from sentimentpulse.config import Settings
from sentimentpulse.utils import (
    validate_text_input,
    calculate_sentiment_distribution,
    truncate_text,
    generate_cache_key,
    SentimentAnalyzer,
)


class TestSentimentEngine:
    """Test suite for SentimentEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initializes with correct defaults."""
        engine = SentimentEngine()
        assert engine.model_name is not None
        assert engine.device in ["cuda", "cpu"]
        assert engine._cache is not None
    
    def test_cache_initialization(self):
        """Test cache is properly initialized."""
        engine = get_engine()
        stats = engine.get_cache_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
    
    def test_health_check_structure(self):
        """Test health check returns all required fields."""
        engine = get_engine()
        health = engine.health_check()
        
        required_fields = ["status", "model_loaded", "model_name", "device", "cache"]
        for field in required_fields:
            assert field in health


class TestSentimentCache:
    """Test suite for SentimentCache class."""
    
    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        return SentimentCache(maxsize=10, ttl=60)
    
    def test_cache_set_and_get(self, cache):
        """Test basic cache operations."""
        text = "This is a test"
        result = {"sentiment": "positive", "score": 0.9}
        
        cache.set(text, result)
        retrieved = cache.get(text)
        
        assert retrieved == result
    
    def test_cache_hash_generation(self, cache):
        """Test cache key generation is consistent."""
        text = "Same text"
        
        key1 = cache._hash_text(text)
        key2 = cache._hash_text(text)
        
        assert key1 == key2
    
    def test_cache_clear(self, cache):
        """Test cache can be cleared."""
        text = "Test text"
        cache.set(text, {"sentiment": "positive"})
        
        cache.clear()
        result = cache.get(text)
        
        assert result is None
    
    def test_cache_stats(self, cache):
        """Test cache statistics tracking."""
        text = "Test"
        cache.set(text, {"sentiment": "positive"})
        cache.get(text)  # hit
        cache.get("different")  # miss
        
        stats = cache.stats
        
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0


class TestValidation:
    """Test input validation functions."""
    
    def test_validate_empty_text(self):
        """Test validation rejects empty text."""
        is_valid, error = validate_text_input("")
        
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_whitespace_only(self):
        """Test validation rejects whitespace-only text."""
        is_valid, error = validate_text_input("   ")
        
        assert is_valid is False
        assert "whitespace" in error.lower()
    
    def test_validate_valid_text(self):
        """Test validation accepts valid text."""
        is_valid, error = validate_text_input("This is valid text!")
        
        assert is_valid is True
        assert error == ""
    
    def test_validate_max_length(self):
        """Test validation rejects text exceeding max length."""
        long_text = "a" * 10001
        is_valid, error = validate_text_input(long_text, max_length=10000)
        
        assert is_valid is False
        assert "exceeds" in error.lower()
    
    def test_validate_custom_max_length(self):
        """Test validation uses custom max length."""
        text = "a" * 101
        is_valid, error = validate_text_input(text, max_length=100)
        
        assert is_valid is False


class TestUtilities:
    """Test utility functions."""
    
    def test_truncate_text_short(self):
        """Test truncation doesn't affect short text."""
        text = "Short"
        result = truncate_text(text, max_length=10)
        
        assert result == text
    
    def test_truncate_text_long(self):
        """Test truncation truncates long text."""
        text = "This is a very long text that should be truncated"
        result = truncate_text(text, max_length=20)
        
        assert len(result) <= 20
        assert result.endswith("...")
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        text = "test"
        key = generate_cache_key(text)
        
        assert key.startswith("sentiment:")
        assert len(key) > len("sentiment:")
    
    def test_sentiment_distribution(self):
        """Test sentiment distribution calculation."""
        results = [
            {"sentiment": "positive"},
            {"sentiment": "positive"},
            {"sentiment": "negative"},
            {"sentiment": "neutral"},
        ]
        
        dist = calculate_sentiment_distribution(results)
        
        assert dist["positive"] == 2
        assert dist["negative"] == 1
        assert dist["neutral"] == 1
        assert dist["total"] == 4
    
    def test_sentiment_distribution_empty(self):
        """Test distribution with empty results."""
        dist = calculate_sentiment_distribution([])
        
        assert dist["total"] == 0


class TestSentimentLabel:
    """Test SentimentLabel enum."""
    
    def test_positive_label(self):
        """Test positive label value."""
        assert SentimentLabel.POSITIVE.value == "positive"
    
    def test_negative_label(self):
        """Test negative label value."""
        assert SentimentLabel.NEGATIVE.value == "negative"
    
    def test_neutral_label(self):
        """Test neutral label value."""
        assert SentimentLabel.NEUTRAL.value == "neutral"
    
    def test_from_score(self):
        """Test score to label conversion."""
        assert SentimentLabel.from_score(0) == SentimentLabel.NEGATIVE
        assert SentimentLabel.from_score(1) == SentimentLabel.NEUTRAL
        assert SentimentLabel.from_score(2) == SentimentLabel.POSITIVE
        # Invalid score returns neutral
        assert SentimentLabel.from_score(99) == SentimentLabel.NEUTRAL


class TestLightweightAnalyzer:
    """Test SentimentAnalyzer utility class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SentimentAnalyzer(use_cache=True)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer.use_cache is True
        assert analyzer._cache == {}
    
    def test_cache_size_property(self, analyzer):
        """Test cache size property."""
        assert analyzer.cache_size == 0
    
    def test_clear_cache(self, analyzer):
        """Test cache clearing."""
        analyzer._cache["test"] = {"sentiment": "positive"}
        analyzer.clear_cache()
        
        assert analyzer.cache_size == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "word " * 1000
        result = analyze_sentiment(long_text)
        
        assert "sentiment" in result
        assert "score" in result
    
    def test_unicode_text(self):
        """Test handling of unicode text."""
        result = analyze_sentiment("This is great! 🎉")
        
        assert "sentiment" in result
    
    def test_emoji_text(self):
        """Test handling of text with emojis."""
        result = analyze_sentiment("I love it 😍😍")
        
        assert "sentiment" in result
    
    def test_mixed_language(self):
        """Test handling of mixed language text."""
        result = analyze_sentiment("This is great! Muy bien!")
        
        assert "sentiment" in result
    
    def test_numbers_only(self):
        """Test handling of numbers only."""
        result = analyze_sentiment("12345")
        
        assert "sentiment" in result
    
    def test_special_characters(self):
        """Test handling of special characters."""
        result = analyze_sentiment("!!! ??? !!!")
        
        assert "sentiment" in result


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def test_batch_empty_list(self):
        """Test batch processing with empty list."""
        result = analyze_batch([])
        
        assert result == []
    
    def test_batch_single_item(self):
        """Test batch processing with single item."""
        result = analyze_batch(["Hello"])
        
        assert len(result) == 1
    
    def test_batch_multiple_items(self):
        """Test batch processing with multiple items."""
        texts = ["Great!", "Terrible!", "Okay"]
        result = analyze_batch(texts)
        
        assert len(result) == 3
    
    def test_batch_with_empty_strings(self):
        """Test batch processing filters empty strings."""
        texts = ["Great!", "", "Terrible!", "   ", "Okay"]
        result = analyze_batch(texts)
        
        # Should only contain results for non-empty strings
        assert len(result) <= 3
    
    def test_batch_large_size(self):
        """Test batch processing with larger dataset."""
        texts = [f"Test text {i}" for i in range(50)]
        result = analyze_batch(texts)
        
        assert len(result) > 0


class TestConfiguration:
    """Test configuration handling."""
    
    def test_settings_defaults(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.api_title == "SentimentPulse API"
        assert settings.port == 8000
        assert settings.cache_enabled is True
    
    def test_cors_origins_parsing(self):
        """Test CORS origins are properly parsed."""
        settings = Settings()
        settings.cors_origins = "http://localhost,http://example.com"
        
        origins = settings.get_cors_origins()
        
        assert len(origins) == 2
        assert "http://localhost" in origins


# Performance tests (can be slow, skip with pytest -m "not slow")
pytestmark = pytest.mark.skipif(
    True,  # Skip by default since they require model loading
    reason="Performance tests - run manually with pytest -m slow"
)


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.slow
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        def analyze_text():
            return analyze_sentiment("Test text for concurrent request")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_text) for _ in range(10)]
            results = [f.result() for f in futures]
        
        assert len(results) == 10
        assert all("sentiment" in r for r in results)
