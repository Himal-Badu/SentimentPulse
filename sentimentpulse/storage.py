"""
SentimentPulse - Database models and persistence
Built by Himal Badu, AI Founder
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


class SentimentType(str, Enum):
    """Sentiment types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class AnalysisRecord:
    """Record of a sentiment analysis."""
    id: Optional[int] = None
    text: str = ""
    sentiment: SentimentType = SentimentType.NEUTRAL
    score: float = 0.0
    confidence: float = 0.0
    model_name: str = ""
    analyzed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class BatchAnalysisRecord:
    """Record of a batch analysis session."""
    id: Optional[int] = None
    texts: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_texts: int = 0
    processing_time_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    user_id: Optional[str] = None


class HistoryManager:
    """Simple in-memory history manager for analysis records."""
    
    def __init__(self, max_records: int = 1000):
        self._records: List[AnalysisRecord] = []
        self._max_records = max_records
    
    def add_record(self, record: AnalysisRecord):
        """Add analysis record to history."""
        self._records.append(record)
        
        # Keep only last max_records
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]
    
    def get_recent(self, limit: int = 10) -> List[AnalysisRecord]:
        """Get recent analysis records."""
        return self._records[-limit:]
    
    def get_by_sentiment(self, sentiment: SentimentType) -> List[AnalysisRecord]:
        """Get records by sentiment type."""
        return [r for r in self._records if r.sentiment == sentiment]
    
    def get_by_user(self, user_id: str) -> List[AnalysisRecord]:
        """Get records by user ID."""
        return [r for r in self._records if r.user_id == user_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about analyzed texts."""
        if not self._records:
            return {
                "total_analyses": 0,
                "by_sentiment": {},
                "average_confidence": 0.0
            }
        
        sentiments = [r.sentiment.value for r in self._records]
        
        return {
            "total_analyses": len(self._records),
            "by_sentiment": {
                "positive": sentiments.count("positive"),
                "negative": sentiments.count("negative"),
                "neutral": sentiments.count("neutral")
            },
            "average_confidence": sum(r.confidence for r in self._records) / len(self._records)
        }
    
    def clear(self):
        """Clear all records."""
        self._records.clear()


# Global history manager
_history_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """Get or create the global history manager."""
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager()
    return _history_manager
