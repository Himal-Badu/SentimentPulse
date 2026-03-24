"""
SentimentPulse - Test configuration
Built by Himal Badu, AI Founder

Pytest configuration and fixtures for testing.
"""

import pytest
from typing import Generator

from sentimentpulse import SentimentEngine


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing."""
    return [
        "I absolutely love this product! It's amazing!",
        "This is terrible. Worst purchase ever.",
        "It's okay, nothing special.",
        "Great quality and fast shipping!",
        "Very disappointed with this item.",
        "Average experience, not great not terrible."
    ]


@pytest.fixture(scope="session")
def positive_texts():
    """Sample positive texts."""
    return [
        "I love this so much!",
        "Amazing quality, highly recommend!",
        "Perfect in every way!",
        "Great product, worth the money!",
        "Exceeded my expectations!"
    ]


@pytest.fixture(scope="session")
def negative_texts():
    """Sample negative texts."""
    return [
        "This is awful. Hate it!",
        "Terrible quality, total waste.",
        "Very disappointed, would not buy again.",
        "Worst product I've ever owned.",
        "Do not recommend to anyone."
    ]


@pytest.fixture(scope="session")
def neutral_texts():
    """Sample neutral texts."""
    return [
        "It is a product.",
        "The item arrived.",
        "Standard quality.",
        "Nothing special.",
        "Average."
    ]


@pytest.fixture(scope="function")
def engine() -> Generator[SentimentEngine, None, None]:
    """Create a fresh engine for each test."""
    eng = SentimentEngine()
    yield eng
    # Cleanup
    eng._cache.clear()


@pytest.fixture(scope="function")
def loaded_engine(engine: SentimentEngine) -> SentimentEngine:
    """Create and load an engine for each test."""
    try:
        engine.load_model()
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    return engine


@pytest.fixture
def mock_result():
    """Mock analysis result for testing."""
    return {
        "sentiment": "positive",
        "score": 0.95,
        "confidence": 0.98,
        "model": "test-model",
        "analyzed_at": "2026-03-24T00:00:00"
    }
