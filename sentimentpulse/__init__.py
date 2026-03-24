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

from sentimentpulse.config import (
    Settings,
    get_settings,
)

from sentimentpulse.utils import (
    validate_text_input,
    calculate_sentiment_distribution,
    truncate_text,
    generate_cache_key,
    SentimentAnalyzer,
)

from sentimentpulse.monitoring import (
    HealthMonitor,
    get_health_monitor,
    SystemMetrics,
    APIMetrics,
)

from sentimentpulse.storage import (
    HistoryManager,
    get_history_manager,
    AnalysisRecord,
    SentimentType,
)

__version__ = "2.0.0"
__author__ = "Himal Badu"

__all__ = [
    # Engine
    "SentimentEngine",
    "get_engine",
    "analyze_sentiment",
    "analyze_batch",
    "SentimentLabel",
    "SentimentPulseError",
    "ModelLoadError",
    "AnalysisError",
    "RateLimitError",
    # Config
    "Settings",
    "get_settings",
    # Utils
    "validate_text_input",
    "calculate_sentiment_distribution",
    "truncate_text",
    "generate_cache_key",
    "SentimentAnalyzer",
    # Monitoring
    "HealthMonitor",
    "get_health_monitor",
    "SystemMetrics",
    "APIMetrics",
    # Storage
    "HistoryManager",
    "get_history_manager",
    "AnalysisRecord",
    "SentimentType",
    # Metadata
    "__version__",
    "__author__",
]
