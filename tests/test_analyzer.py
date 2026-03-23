"""
Tests for SentimentPulse Analyzer
Built by Himal Badu, AI Founder
"""

import pytest
from sentimentpulse import SentimentAnalyzer, analyze_sentiment, analyze_batch


def test_analyzer_positive():
    """Test positive sentiment detection."""
    result = analyze_sentiment("This is amazing and wonderful!")
    assert result['sentiment'] == 'positive'
    assert result['score'] > 0


def test_analyzer_negative():
    """Test negative sentiment detection."""
    result = analyze_sentiment("This is terrible and awful")
    assert result['sentiment'] == 'negative'
    assert result['score'] < 0


def test_analyzer_neutral():
    """Test neutral sentiment detection."""
    result = analyze_sentiment("The meeting is at 3pm")
    assert result['sentiment'] == 'neutral'


def test_confidence_range():
    """Test confidence is between 0 and 1."""
    result = analyze_sentiment("Great product!")
    assert 0 <= result['confidence'] <= 1


def test_batch_analysis():
    """Test batch processing."""
    texts = ["I love this", "I hate this", "It's okay"]
    results = analyze_batch(texts)
    
    assert len(results) == 3
    assert all('sentiment' in r for r in results)


def test_verbose_mode():
    """Test verbose output includes raw scores."""
    result = analyze_sentiment("Great!", verbose=True)
    assert 'raw_scores' in result
    assert 'pos' in result['raw_scores']


def test_empty_text():
    """Test handling of empty text."""
    result = analyze_sentiment("")
    assert result['sentiment'] == 'neutral'
    assert result['score'] == 0
