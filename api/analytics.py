"""
SentimentPulse - Additional API endpoints for analytics
Built by Himal Badu, AI Founder
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from api.models import AnalyzeResponse
from sentimentpulse import get_engine
from sentimentpulse.monitoring import get_health_monitor


# Create router
analytics_router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@analytics_router.get("/summary")
async def get_analytics_summary():
    """Get overall analytics summary."""
    engine = get_engine()
    monitor = get_health_monitor()
    
    # Get cache stats
    cache_stats = engine.get_cache_stats()
    
    # Get health metrics
    health = monitor.get_health_status()
    
    return {
        "summary": {
            "cache": cache_stats,
            "health": health,
            "model": {
                "name": engine.model_name,
                "device": engine.device
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@analytics_router.get("/distribution")
async def get_sentiment_distribution(
    sample_size: int = Query(100, ge=1, le=1000, description="Number of recent results to analyze")
):
    """Get sentiment distribution from recent analyses.
    
    Note: This requires tracking analysis history.
    """
    # For now, return a placeholder
    # In production, you'd track analysis results in a database
    return {
        "message": "Distribution tracking coming soon",
        "sample_size": sample_size,
        "distribution": {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
    }


@analytics_router.get("/performance")
async def get_performance_metrics():
    """Get API performance metrics."""
    monitor = get_health_monitor()
    metrics = monitor.get_api_metrics()
    
    return {
        "performance": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }


@analytics_router.get("/system")
async def get_system_metrics():
    """Get system resource metrics."""
    monitor = get_health_monitor()
    system = monitor.get_system_metrics()
    
    return {
        "cpu_percent": system.cpu_percent,
        "memory_percent": system.memory_percent,
        "memory_used_mb": system.memory_used_mb,
        "memory_available_mb": system.memory_available_mb,
        "disk_used_percent": system.disk_used_percent,
        "timestamp": system.timestamp
    }
