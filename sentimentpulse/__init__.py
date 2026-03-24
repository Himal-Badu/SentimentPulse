# SentimentPulse
# Built by Himal Badu, AI Founder
# Main entry point - re-exports all public APIs

from sentimentpulse.engine import (
    SentimentEngine,
    get_engine,
    analyze_sentiment,
    analyze_batch,
    SentimentLabel,
    SentimentPulseError,
    ModelLoadError,
    AnalysisError,
    RateLimitError,
)

__version__ = "2.0.0"
__author__ = "Himal Badu"

__all__ = [
    "SentimentEngine",
    "get_engine",
    "analyze_sentiment",
    "analyze_batch",
    "SentimentLabel",
    "SentimentPulseError",
    "ModelLoadError",
    "AnalysisError",
    "RateLimitError",
    "__version__",
]
