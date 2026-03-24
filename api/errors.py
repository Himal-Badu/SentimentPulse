"""
SentimentPulse - Error handling and custom exceptions
Built by Himal Badu, AI Founder

Custom exceptions and error handlers for the API.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import Request, status
from fastapi.responses import JSONResponse
from loguru import logger


class SentimentPulseException(Exception):
    """Base exception for SentimentPulse errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelException(SentimentPulseException):
    """Exception for model-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            status_code=500,
            details=details
        )


class ModelLoadException(ModelException):
    """Exception when model fails to load."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Failed to load model: {model_name}",
            details=details
        )


class ModelNotLoadedException(ModelException):
    """Exception when model is not loaded."""
    
    def __init__(self):
        super().__init__(
            message="Model not loaded. Please wait for model to load or restart the service.",
            error_code="MODEL_NOT_LOADED",
            status_code=503
        )


class ValidationException(SentimentPulseException):
    """Exception for input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class TextTooLongException(ValidationException):
    """Exception when input text exceeds maximum length."""
    
    def __init__(self, text_length: int, max_length: int):
        super().__init__(
            message=f"Text length {text_length} exceeds maximum {max_length} characters",
            field="text"
        )


class EmptyTextException(ValidationException):
    """Exception when input text is empty."""
    
    def __init__(self):
        super().__init__(
            message="Text cannot be empty",
            field="text"
        )


class CacheException(SentimentPulseException):
    """Exception for cache-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details=details
        )


class RateLimitException(SentimentPulseException):
    """Exception when rate limit is exceeded."""
    
    def __init__(
        self,
        limit: int,
        window: str,
        retry_after: Optional[int] = None
    ):
        details = {
            "limit": limit,
            "window": window,
            "retry_after": retry_after
        }
        super().__init__(
            message=f"Rate limit exceeded. Maximum {limit} requests per {window}.",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details
        )


class AnalysisException(SentimentPulseException):
    """Exception during sentiment analysis."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ANALYSIS_ERROR",
            status_code=500,
            details=details
        )


# Error handler functions
async def sentimentpulse_exception_handler(
    request: Request,
    exc: SentimentPulseException
) -> JSONResponse:
    """Handle SentimentPulse exceptions."""
    logger.error(
        f"SentimentPulse exception: {exc.error_code} - {exc.message}"
    )
    
    content = {
        "error": exc.error_code,
        "message": exc.message,
        "timestamp": datetime.utcnow().isoformat(),
        "path": str(request.url.path)
    }
    
    if exc.details:
        content["details"] = exc.details
    
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        content["request_id"] = request_id
    
    return JSONResponse(
        status_code=exc.status_code,
        content=content
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    
    content = {
        "error": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred",
        "timestamp": datetime.utcnow().isoformat(),
        "path": str(request.url.path)
    }
    
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        content["request_id"] = request_id
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content
    )


def create_error_response(
    error: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response dictionary."""
    response = {
        "error": error,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if details:
        response["details"] = details
    
    if request_id:
        response["request_id"] = request_id
    
    return response
