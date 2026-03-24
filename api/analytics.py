"""
SentimentPulse - Advanced Analytics API
Built by Himal Badu, AI Founder

Provides analytics endpoints for sentiment analysis data.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from sentimentpulse import get_engine
from sentimentpulse.utils import calculate_sentiment_distribution


router = APIRouter()


class TrendDataPoint(BaseModel):
    """Data point for trend analysis."""
    timestamp: str
    positive: int
    negative: int
    neutral: int
    total: int


class AnalyticsSummary(BaseModel):
    """Analytics summary response."""
    total_analyses: int
    sentiment_distribution: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    average_confidence: float
    time_range: Dict[str, str]


# In-memory analytics storage
class AnalyticsStore:
    """Store for analytics data."""
    
    def __init__(self):
        self.analyses: List[Dict[str, Any]] = []
        self.max_records = 10000
    
    def record_analysis(self, result: Dict[str, Any], text: str = ""):
        """Record an analysis for analytics."""
        self.analyses.append({
            "result": result,
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only recent records
        if len(self.analyses) > self.max_records:
            self.analyses = self.analyses[-self.max_records:]
    
    def get_analyses(
        self,
        limit: int = 100,
        sentiment: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recorded analyses."""
        analyses = self.analyses
        
        if sentiment:
            analyses = [
                a for a in analyses 
                if a["result"].get("sentiment") == sentiment
            ]
        
        return analyses[-limit:]
    
    def get_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get analytics summary for time range."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        recent = [
            a for a in self.analyses
            if datetime.fromisoformat(a["timestamp"]) > cutoff
        ]
        
        if not recent:
            return {
                "total_analyses": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "sentiment_percentages": {"positive": 0, "negative": 0, "neutral": 0},
                "average_confidence": 0.0,
                "time_range": {
                    "start": cutoff.isoformat(),
                    "end": datetime.utcnow().isoformat()
                }
            }
        
        results = [a["result"] for a in recent]
        distribution = calculate_sentiment_distribution(results)
        
        confidences = [r.get("confidence", 0) for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_analyses": len(recent),
            "sentiment_distribution": {
                "positive": distribution["positive"],
                "negative": distribution["negative"],
                "neutral": distribution["neutral"]
            },
            "sentiment_percentages": {
                "positive": distribution["positive_pct"],
                "negative": distribution["negative_pct"],
                "neutral": distribution["neutral_pct"]
            },
            "average_confidence": round(avg_confidence, 4),
            "time_range": {
                "start": cutoff.isoformat(),
                "end": datetime.utcnow().isoformat()
            }
        }
    
    def get_trends(self, hours: int = 24, interval_minutes: int = 60) -> List[TrendDataPoint]:
        """Get trend data for specified time range."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Group analyses by time interval
        intervals = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
        
        for analysis in self.analyses:
            timestamp = datetime.fromisoformat(analysis["timestamp"])
            
            if timestamp < cutoff:
                continue
            
            # Calculate interval key
            interval_start = timestamp.replace(
                minute=(timestamp.minute // interval_minutes) * interval_minutes,
                second=0,
                microsecond=0
            )
            key = interval_start.isoformat()
            
            sentiment = analysis["result"].get("sentiment", "neutral")
            intervals[key][sentiment] = intervals[key].get(sentiment, 0) + 1
            intervals[key]["total"] += 1
        
        # Convert to list of data points
        trends = [
            TrendDataPoint(
                timestamp=key,
                positive=data["positive"],
                negative=data["negative"],
                neutral=data["neutral"],
                total=data["total"]
            )
            for key, data in sorted(intervals.items())
        ]
        
        return trends
    
    def get_top_sentiments(
        self,
        limit: int = 10,
        sentiment: str = "positive"
    ) -> List[Dict[str, Any]]:
        """Get most positive or negative texts."""
        sorted_analyses = sorted(
            self.analyses,
            key=lambda x: x["result"].get("score", 0),
            reverse=(sentiment == "positive")
        )
        
        results = []
        for analysis in sorted_analyses[:limit]:
            if analysis["result"].get("sentiment") == sentiment:
                results.append({
                    "text": analysis["text"][:100] + "..." if len(analysis["text"]) > 100 else analysis["text"],
                    "score": analysis["result"].get("score"),
                    "confidence": analysis["result"].get("confidence"),
                    "timestamp": analysis["timestamp"]
                })
        
        return results
    
    def clear(self):
        """Clear all analytics data."""
        self.analyses.clear()


# Global analytics store
_analytics_store = AnalyticsStore()


@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to analyze")
):
    """Get analytics summary for the specified time range."""
    return _analytics_store.get_summary(hours)


@router.get("/trends")
async def get_trends(
    hours: int = Query(default=24, ge=1, le=168),
    interval: int = Query(default=60, ge=5, le=360, description="Interval in minutes")
):
    """Get sentiment trends over time."""
    trends = _analytics_store.get_trends(hours, interval)
    return {
        "trends": trends,
        "interval_minutes": interval,
        "hours": hours
    }


@router.get("/recent")
async def get_recent_analyses(
    limit: int = Query(default=50, ge=1, le=500),
    sentiment: Optional[str] = Query(default=None)
):
    """Get recent analyses with optional filtering."""
    analyses = _analytics_store.get_analyses(limit, sentiment)
    return {
        "analyses": analyses,
        "count": len(analyses)
    }


@router.get("/top/{sentiment_type}")
async def get_top_sentiments(
    sentiment_type: str,
    limit: int = Query(default=10, ge=1, le=100)
):
    """Get most positive or negative texts."""
    if sentiment_type not in ["positive", "negative"]:
        raise HTTPException(status_code=400, detail="Invalid sentiment type")
    
    top = _analytics_store.get_top_sentiments(limit, sentiment_type)
    return {
        "sentiment_type": sentiment_type,
        "results": top
    }


@router.post("/record")
async def record_analysis(data: Dict[str, Any]):
    """Record an analysis for analytics."""
    result = data.get("result", {})
    text = data.get("text", "")
    
    _analytics_store.record_analysis(result, text)
    
    return {"status": "recorded"}


@router.delete("/clear")
async def clear_analytics():
    """Clear all analytics data."""
    _analytics_store.clear()
    return {"status": "cleared"}


def get_analytics_store() -> AnalyticsStore:
    """Get the global analytics store."""
    return _analytics_store
