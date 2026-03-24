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
import time
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
from loguru import logger

from sentimentpulse import get_engine
from sentimentpulse.engine import SentimentEngine
from api.models import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeResponse,
    HealthResponse,
    ErrorResponse,
    RootResponse,
    CacheStatsResponse,
    CacheClearResponse,
    ModelInfo,
    APIInfo,
)


# ============================================================================
# Configuration
# ============================================================================

# API Configuration
API_TITLE = os.getenv("API_TITLE", "SentimentPulse API")
API_VERSION = os.getenv("API_VERSION", "2.0.0")
API_DESCRIPTION = os.getenv(
    "API_DESCRIPTION",
    "Production-grade sentiment analysis API powered by transformers"
)

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Rate Limiting
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
BATCH_RATE_LIMIT_PER_MINUTE = int(os.getenv("BATCH_RATE_LIMIT_PER_MINUTE", "20"))

# Caching
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


# ============================================================================
# Rate Limiting Setup
# ============================================================================

def get_client_ip(request: Request) -> str:
    """Get client IP for rate limiting."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=get_client_ip)


# ============================================================================
# Monitoring Setup
# ============================================================================

def setup_monitoring():
    """Setup monitoring and error tracking."""
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.loguru import LoguruIntegration
    
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[
                FastApiIntegration(),
                LoguruIntegration(),
            ],
            traces_sample_rate=0.1,
            environment=os.getenv("ENVIRONMENT", "production"),
        )
        logger.info("Sentry monitoring enabled")


# ============================================================================
# API Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting SentimentPulse API...")
    
    if os.getenv("SENTRY_DSN"):
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
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
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
    allow_origins=CORS_ORIGINS,
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

@app.get("/", response_model=RootResponse, tags=["Root"])
async def root():
    """Root endpoint."""
    return RootResponse(
        name="SentimentPulse API",
        version=API_VERSION,
        docs="/docs",
        health="/health"
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_engine()
        health = engine.health_check()
        
        return HealthResponse(
            status="healthy",
            version=API_VERSION,
            model_loaded=health["model_loaded"],
            model_name=health["model_name"],
            cache_stats=health["cache"] if CACHE_ENABLED else None
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.get("/api/v1/model", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Get model information."""
    try:
        engine = get_engine()
        health = engine.health_check()
        
        return ModelInfo(
            model_name=health["model_name"],
            model_type="RoBERTa",
            max_length=512,
            device=health["device"],
            loaded=health["model_loaded"]
        )
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/info", response_model=APIInfo, tags=["Info"])
async def api_info():
    """Get API information."""
    return APIInfo(
        name=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        documentation="https://github.com/Himal-Badu/SentimentPulse#readme",
        repository="https://github.com/Himal-Badu/SentimentPulse"
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute" if RATE_LIMIT_ENABLED else "99999/minute")
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
@limiter.limit(f"{BATCH_RATE_LIMIT_PER_MINUTE}/minute" if RATE_LIMIT_ENABLED else "99999/minute")
async def analyze_batch_texts(request: Request, body: BatchAnalyzeRequest):
    """Analyze sentiment of multiple texts."""
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
            processing_time_ms=round(process_time, 2),
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/cache/stats", response_model=CacheStatsResponse, tags=["Cache"])
async def cache_stats():
    """Get cache statistics."""
    if not CACHE_ENABLED:
        return CacheStatsResponse(
            hits=0,
            misses=0,
            size=0,
            hit_rate_percent=0.0
        )
    
    engine = get_engine()
    stats = engine.get_cache_stats()
    
    return CacheStatsResponse(**stats)


@app.delete("/api/v1/cache", response_model=CacheClearResponse, tags=["Cache"])
async def clear_cache():
    """Clear the sentiment analysis cache."""
    if not CACHE_ENABLED:
        return CacheClearResponse(
            message="Cache is disabled",
            previous_size=None
        )
    
    engine = get_engine()
    previous_size = len(engine._cache._cache)
    
    engine._cache.clear()
    
    return CacheClearResponse(
        message="Cache cleared successfully",
        previous_size=previous_size
    )


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
        host=HOST,
        port=PORT,
        reload=os.getenv("ENVIRONMENT") == "development",
        workers=int(os.getenv("WORKERS", "1")),
        log_level="info"
    )
