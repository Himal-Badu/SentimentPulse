"""
Tests for SentimentPulse Analyzer
Built by Himal Badu, AI Founder
"""

import pytest
from sentimentpulse import analyze_sentiment, analyze_batch, get_engine


class TestSentimentAnalyzer:
    """Test suite for sentiment analysis engine."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.engine = get_engine()
        self.engine.load_model()
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        result = analyze_sentiment("This is amazing and wonderful!")
        assert result['sentiment'] == 'positive'
        assert result['score'] > 0
        assert 0 <= result['confidence'] <= 1
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        result = analyze_sentiment("This is terrible and awful")
        assert result['sentiment'] == 'negative'
        assert result['score'] < 0
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        result = analyze_sentiment("The meeting is at 3pm")
        assert result['sentiment'] in ['neutral', 'positive', 'negative']
    
    def test_confidence_range(self):
        """Test confidence is between 0 and 1."""
        result = analyze_sentiment("Great product!")
        assert 0 <= result['confidence'] <= 1
    
    def test_batch_analysis(self):
        """Test batch processing."""
        texts = ["I love this", "I hate this", "It's okay"]
        results = analyze_batch(texts)
        
        assert len(results) == 3
        assert all('sentiment' in r for r in results)
    
    def test_verbose_mode(self):
        """Test verbose output includes raw scores."""
        result = analyze_sentiment("Great!", verbose=True)
        assert 'raw_scores' in result
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = analyze_sentiment("")
        assert result['sentiment'] in ['neutral', 'positive', 'negative']
    
    def test_whitespace_only(self):
        """Test handling of whitespace-only text."""
        result = analyze_sentiment("   ")
        assert result['sentiment'] in ['neutral', 'positive', 'negative']
    
    def test_model_name_in_response(self):
        """Test that model name is included in response."""
        result = analyze_sentiment("Hello world")
        assert 'model' in result
    
    def test_timestamp_in_response(self):
        """Test that timestamp is included in response."""
        result = analyze_sentiment("Hello world")
        assert 'analyzed_at' in result
    
    def test_caching(self):
        """Test that caching works."""
        text = "This is a test for caching"
        
        # First call - should be cache miss
        result1 = analyze_sentiment(text)
        
        # Second call - should be cache hit
        result2 = analyze_sentiment(text)
        
        assert result1 == result2


class TestEngine:
    """Test suite for SentimentEngine."""
    
    def test_health_check(self):
        """Test health check returns proper structure."""
        engine = get_engine()
        engine.load_model()
        health = engine.health_check()
        
        assert 'status' in health
        assert 'model_loaded' in health
        assert 'model_name' in health
    
    def test_cache_stats(self):
        """Test cache statistics."""
        engine = get_engine()
        stats = engine.get_cache_stats()
        
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'size' in stats
