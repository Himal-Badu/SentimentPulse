"""
SentimentPulse API - Production-Grade REST API
Built by Himal Badu, AI Founder

A high-performance FastAPI application with:
- Rate limiting
- Request validation
- Error handling
- Caching
- Monitoring
- OpenAPI documentation
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from loguru import logger
import redis.asyncio as redis

from sentimentpulse import get_engine, analyze_sentiment, analyze_batch


# ============================================================================
# Configuration
# ============================================================================

class AppConfig(BaseSettings):
    """Application configuration."""
    # API Settings
    API_TITLE: str = "SentimentPulse API"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "Production-grade sentiment analysis API powered by transformers"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    REDIS_URL: Optional[str] = None
    
    # Caching
    CACHE_ENABLED: bool = True
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = AppConfig()


# ============================================================================
# Rate Limiting Setup
# ============================================================================

def get_client_ip(request: Request) -> str:
    """Get client IP for rate limiting."""
    # Check for forwarded headers (reverse proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=get_client_ip)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_monitoring():
    """Setup monitoring and error tracking."""
    # Sentry integration
    if config.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.loguru import LoguruIntegration
        
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            integrations=[
                FastApiIntegration(),
                LoguruIntegration(),
            ],
            traces_sample_rate=0.1,
            environment=os.getenv("ENVIRONMENT", "production"),
        )
        logger.info("Sentry monitoring enabled")


# ============================================================================
# Request/Response Models
# ============================================================================

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


class AnalyzeResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str = Field(..., description="Sentiment: positive, negative, or neutral")
    score: float = Field(..., description="Normalized sentiment score (-1 to 1)")
    confidence: float = Field(..., description="Model confidence (0 to 1)")
    model: str = Field(..., description="Model used for analysis")
    analyzed_at: str = Field(..., description="Timestamp of analysis")
    raw_scores: Optional[Dict[str, Any]] = Field(default=None, description="Detailed model scores")


class BatchAnalyzeResponse(BaseModel):
    """Response model for batch analysis."""
    results: List[AnalyzeResponse] = Field(..., description="Analysis results")
    total: int = Field(..., description="Total number of texts analyzed")
    processed_at: str = Field(..., description="Processing timestamp")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    model_loaded: bool
    model_name: str
    cache_stats: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None
    timestamp: str


# ============================================================================
# API Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting SentimentPulse API...")
    setup_monitoring()
    
    # Pre-load the model in background
    import threading
    def load_model():
        try:
            engine = get_engine()
            engine.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    threading.Thread(target=load_model, daemon=True).start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down SentimentPulse API...")


# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.utcnow()
    
    response = await call_next(request)
    
    process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response


# ============================================================================
# API Routes
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "SentimentPulse API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_engine()
        health = engine.health_check()
        
        return HealthResponse(
            status="healthy",
            version=config.API_VERSION,
            model_loaded=health["model_loaded"],
            model_name=health["model_name"],
            cache_stats=health["cache"] if config.CACHE_ENABLED else None
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
@limiter.limit("60/minute" if config.RATE_LIMIT_ENABLED else "99999/minute")
async def analyze_text(request: Request, body: AnalyzeRequest):
    """
    Analyze sentiment of a single text.
    
    Returns sentiment (positive/negative/neutral), score, and confidence.
    Uses transformer-based model for accurate analysis.
    """
    try:
        engine = get_engine()
        
        # Ensure model is loaded
        if not engine._model_loaded:
            engine.load_model()
        
        result = engine.analyze(
            text=body.text,
            use_cache=body.use_cache,
            verbose=body.verbose
        )
        
        return AnalyzeResponse(**result)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Analysis"])
@limiter.limit("20/minute" if config.RATE_LIMIT_ENABLED else "99999/minute")
async def analyze_batch_texts(request: Request, body: BatchAnalyzeRequest):
    """
    Analyze sentiment of multiple texts.
    
    Efficiently processes up to 500 texts in a single request.
    Returns array of results with metadata.
    """
    start_time = datetime.utcnow()
    
    try:
        engine = get_engine()
        
        # Ensure model is loaded
        if not engine._model_loaded:
            engine.load_model()
        
        results = engine.analyze_batch(
            texts=body.texts,
            use_cache=body.use_cache,
            verbose=body.verbose,
            show_progress=False
        )
        
        process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return BatchAnalyzeResponse(
            results=[AnalyzeResponse(**r) for r in results],
            total=len(results),
            processed_at=datetime.utcnow().isoformat(),
            processing_time_ms=round(process_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/cache/stats", tags=["Cache"])
async def cache_stats():
    """Get cache statistics."""
    if not config.CACHE_ENABLED:
        return {"message": "Cache is disabled"}
    
    engine = get_engine()
    return engine.get_cache_stats()


@app.delete("/api/v1/cache", tags=["Cache"])
async def clear_cache():
    """Clear the sentiment analysis cache."""
    if not config.CACHE_ENABLED:
        return {"message": "Cache is disabled"}
    
    engine = get_engine()
    engine._cache.clear()
    return {"message": "Cache cleared successfully"}


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat()
        ).model_dump()
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=os.getenv("ENVIRONMENT") == "development",
        workers=int(os.getenv("WORKERS", "1")),
        log_level="info"
    )
