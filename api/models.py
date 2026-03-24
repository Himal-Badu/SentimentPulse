"""
API models (request/response schemas) for SentimentPulse
Built by Himal Badu, AI Founder

Pydantic models for request validation and response formatting.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, field_serializer
from enum import Enum


class SentimentType(str, Enum):
    """Sentiment types supported by the API."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AnalyzeRequest(BaseModel):
    """Request model for single text analysis."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze for sentiment",
        examples=["I love this product! It's amazing and works perfectly."]
    )
    verbose: bool = Field(
        default=False,
        description="Include detailed model scores in response"
    )
    use_cache: bool = Field(
        default=True,
        description="Use caching for this request"
    )
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "I love this product! It's amazing!",
                    "verbose": False,
                    "use_cache": True
                }
            ]
        }
    }


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch analysis."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of texts to analyze",
        examples=[["I love this!", "This is terrible.", "It's okay."]]
    )
    verbose: bool = Field(
        default=False,
        description="Include detailed model scores in response"
    )
    use_cache: bool = Field(
        default=True,
        description="Use caching for this request"
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Texts list cannot be empty")
        # Filter and validate each text
        return [t.strip() for t in v if t and t.strip()]
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": ["Great product!", "Terrible service", "Average quality"],
                    "verbose": False,
                    "use_cache": True
                }
            ]
        }
    }


class RawScores(BaseModel):
    """Detailed model scores for verbose responses."""
    label: str = Field(..., description="Model label")
    raw_score: float = Field(..., description="Raw model score")


class AnalyzeResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str = Field(..., description="Sentiment: positive, negative, or neutral")
    score: float = Field(..., description="Normalized sentiment score (-1 to 1)")
    confidence: float = Field(..., description="Model confidence (0 to 1)")
    model: str = Field(..., description="Model used for analysis")
    analyzed_at: str = Field(..., description="Timestamp of analysis")
    raw_scores: Optional[RawScores] = Field(default=None, description="Detailed model scores")
    
    @field_serializer('analyzed_at')
    def serialize_datetime(self, dt: Any) -> str:
        if isinstance(dt, datetime):
            return dt.isoformat()
        return str(dt)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sentiment": "positive",
                    "score": 0.9452,
                    "confidence": 0.9823,
                    "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "analyzed_at": "2026-03-23T12:00:00Z"
                }
            ]
        }
    }


class BatchAnalyzeResponse(BaseModel):
    """Response model for batch analysis."""
    results: List[AnalyzeResponse] = Field(..., description="Analysis results")
    total: int = Field(..., description="Total number of texts analyzed")
    processed_at: str = Field(..., description="Processing timestamp")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time")
    cached: bool = Field(default=False, description="Whether results were cached")
    
    @field_serializer('processed_at')
    def serialize_datetime(self, dt: Any) -> str:
        if isinstance(dt, datetime):
            return dt.isoformat()
        return str(dt)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "sentiment": "positive",
                            "score": 0.9452,
                            "confidence": 0.9823,
                            "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                            "analyzed_at": "2026-03-23T12:00:00Z"
                        }
                    ],
                    "total": 1,
                    "processed_at": "2026-03-23T12:00:00Z",
                    "processing_time_ms": 125.5,
                    "cached": False
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Model name")
    cache_stats: Optional[Dict[str, Any]] = Field(default=None, description="Cache statistics")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "2.0.0",
                    "model_loaded": True,
                    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "cache_stats": {
                        "hits": 150,
                        "misses": 50,
                        "size": 200,
                        "hit_rate_percent": 75.0
                    }
                }
            ]
        }
    }


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    hits: int = Field(..., description="Number of cache hits")
    misses: int = Field(..., description="Number of cache misses")
    size: int = Field(..., description="Current cache size")
    hit_rate_percent: float = Field(..., description="Cache hit rate percentage")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hits": 150,
                    "misses": 50,
                    "size": 200,
                    "hit_rate_percent": 75.0
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error type")
    detail: Optional[str] = Field(default=None, description="Detailed error message")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracing")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "AnalysisError",
                    "detail": "Failed to analyze text: invalid input",
                    "timestamp": "2026-03-23T12:00:00Z",
                    "request_id": "abc123"
                }
            ]
        }
    }


class RootResponse(BaseModel):
    """Response model for root endpoint."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    docs: str = Field(..., description="Documentation URL")
    health: str = Field(..., description="Health check URL")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "SentimentPulse API",
                    "version": "2.0.0",
                    "docs": "/docs",
                    "health": "/health"
                }
            ]
        }
    }


class CacheClearResponse(BaseModel):
    """Response model for cache clear operation."""
    message: str = Field(..., description="Operation result message")
    previous_size: Optional[int] = Field(default=None, description="Previous cache size")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Cache cleared successfully",
                    "previous_size": 150
                }
            ]
        }
    }


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    max_length: int = Field(..., description="Maximum input length")
    device: str = Field(..., description="Device used for inference")
    loaded: bool = Field(..., description="Whether model is loaded")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "model_type": "RoBERTa",
                    "max_length": 512,
                    "device": "cpu",
                    "loaded": True
                }
            ]
        }
    }


class APIInfo(BaseModel):
    """API information response."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    documentation: str = Field(..., description="Documentation URL")
    repository: str = Field(..., description="Repository URL")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "SentimentPulse API",
                    "version": "2.0.0",
                    "description": "Production-grade sentiment analysis API",
                    "documentation": "https://github.com/Himal-Badu/SentimentPulse#readme",
                    "repository": "https://github.com/Himal-Badu/SentimentPulse"
                }
            ]
        }
    }
