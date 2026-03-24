"""
Middleware and utilities for SentimentPulse API
Built by Himal Badu, AI Founder
"""

import time
import uuid
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from loguru import logger
import hashlib


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add unique request ID to each request."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to measure request processing time."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.log_bodies = False  # Don't log request bodies by default
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log incoming request
        logger.info(
            f"Request started | {request.method} {request.url.path} | "
            f"ID: {request_id} | Client: {request.client.host if request.client else 'unknown'}"
        )
        
        response = await call_next(request)
        
        # Log response
        logger.info(
            f"Request completed | {request.method} {request.url.path} | "
            f"Status: {response.status_code} | ID: {request_id}"
        )
        
        return response


def generate_cache_key(text: str, prefix: str = "sentiment") -> str:
    """Generate a cache key from text.
    
    Args:
        text: Input text
        prefix: Prefix for the cache key
        
    Returns:
        Cache key string
    """
    hash_obj = hashlib.sha256(text.encode())
    return f"{prefix}:{hash_obj.hexdigest()[:16]}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_batch_response(
    results: list,
    processing_time_ms: float,
    cached: bool = False
) -> Dict[str, Any]:
    """Format batch analysis response.
    
    Args:
        results: List of analysis results
        processing_time_ms: Time taken to process
        cached: Whether results were from cache
        
    Returns:
        Formatted response dictionary
    """
    return {
        "results": results,
        "total": len(results),
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_time_ms": round(processing_time_ms, 2),
        "cached": cached
    }


def validate_text_input(text: str, max_length: int = 10000) -> tuple[bool, str]:
    """Validate text input for analysis.
    
    Args:
        text: Input text to validate
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text:
        return False, "Text cannot be empty"
    
    if not text.strip():
        return False, "Text cannot be whitespace only"
    
    if len(text) > max_length:
        return False, f"Text exceeds maximum length of {max_length} characters"
    
    return True, ""


def calculate_sentiment_distribution(results: list) -> Dict[str, Any]:
    """Calculate sentiment distribution from results.
    
    Args:
        results: List of sentiment analysis results
        
    Returns:
        Dictionary with distribution statistics
    """
    total = len(results)
    if total == 0:
        return {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
    
    sentiments = [r.get("sentiment") for r in results]
    
    positive = sentiments.count("positive")
    negative = sentiments.count("negative")
    neutral = sentiments.count("neutral")
    
    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "total": total,
        "positive_pct": round(positive / total * 100, 2),
        "negative_pct": round(negative / total * 100, 2),
        "neutral_pct": round(neutral / total * 100, 2),
    }


def create_error_response(
    error: str,
    detail: str = None,
    status_code: int = 500
) -> JSONResponse:
    """Create standardized error response.
    
    Args:
        error: Error type/name
        detail: Detailed error message
        status_code: HTTP status code
        
    Returns:
        JSONResponse with error details
    """
    content = {
        "error": error,
        "detail": detail,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )


class SentimentAnalyzer:
    """Lightweight sentiment analyzer for simple use cases.
    
    This class provides a simpler interface for basic sentiment analysis
    without the full engine overhead.
    """
    
    def __init__(self, use_cache: bool = True):
        """Initialize the analyzer.
        
        Args:
            use_cache: Whether to enable caching
        """
        self.use_cache = use_cache
        self._cache: Dict[str, Dict] = {}
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment result dictionary
        """
        # Check cache
        if self.use_cache and text in self._cache:
            return self._cache[text]
        
        # Import and use engine
        from sentimentpulse.engine import analyze_sentiment
        result = analyze_sentiment(text, use_cache=self.use_cache)
        
        # Cache result
        if self.use_cache:
            self._cache[text] = result
        
        return result
    
    def analyze_many(self, texts: list) -> list:
        """Analyze multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of results
        """
        from sentimentpulse.engine import analyze_batch
        return analyze_batch(texts, use_cache=self.use_cache)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
