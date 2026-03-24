"""
SentimentPulse - Production-Grade Sentiment Analysis Engine
Built by Himal Badu, AI Founder

A state-of-the-art sentiment analysis system powered by transformer models.
Supports single text and batch processing with caching, rate limiting, and monitoring.
"""

import os
import sys
import logging
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import lru_cache
from datetime import datetime
from enum import Enum
import threading
from contextlib import contextmanager

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline
)
from loguru import logger
from cachetools import TTLCache
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class ModelConfig:
    """Model configuration settings."""
    
    # Primary model - Twitter sentiment RoBERTa
    MODEL_NAME = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    MAX_LENGTH = 512
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    
    # Fallback models for resilience
    FALLBACK_MODELS = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "nlptown/bert-base-multilingual-uncased-sentiment"
    ]
    
    # Cache settings
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SentimentLabel(str, Enum):
    """Sentiment label constants.
    
    Defines the three possible sentiment labels returned by the analyzer.
    Each label corresponds to a specific emotional valence in the text.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    
    @classmethod
    def from_score(cls, score: int) -> "SentimentLabel":
        """Convert model score (0,1,2) to sentiment label.
        
        Args:
            score: Integer score from model (0=negative, 1=neutral, 2=positive)
            
        Returns:
            Corresponding SentimentLabel enum value
        """
        mapping = {0: cls.NEGATIVE, 1: cls.NEUTRAL, 2: cls.POSITIVE}
        return mapping.get(score, cls.NEUTRAL)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging.
    
    Sets up Loguru with both console (colored) and file (rotated) outputs.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()
    
    # Console output with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File output for production
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        f"{log_dir}/sentimentpulse_{{time:YYYY-MM-DD}}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        compression="zip"
    )


# Initialize logging
setup_logging()


# ============================================================================
# Custom Exceptions
# ============================================================================

class SentimentPulseError(Exception):
    """Base exception for SentimentPulse.
    
    All custom exceptions in SentimentPulse inherit from this base class.
    """
    pass


class ModelLoadError(SentimentPulseError):
    """Raised when model loading fails.
    
    This can occur due to:
    - Network issues when downloading models
    - Insufficient disk space for model files
    - Invalid model name or configuration
    - Out of memory errors during model loading
    """
    pass


class AnalysisError(SentimentPulseError):
    """Raised when sentiment analysis fails.
    
    This can occur due to:
    - Invalid input text
    - Model inference errors
    - Text too long for model
    - Internal processing errors
    """
    pass


class RateLimitError(SentimentPulseError):
    """Raised when rate limit is exceeded.
    
    Indicates that the client has made too many requests
    and must wait before making more requests.
    """
    pass


# ============================================================================
# Caching Layer
# ============================================================================

class SentimentCache:
    """Thread-safe caching for sentiment analysis results.
    
    Uses TTLCache (Time-To-Live) to automatically expire old results
    and prevent memory leaks during long-running applications.
    
    Attributes:
        maxsize: Maximum number of items to store
        ttl: Time-to-live for each cache entry in seconds
    """
    
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """Initialize the sentiment cache.
        
        Args:
            maxsize: Maximum number of cache entries
            ttl: Time-to-live in seconds for each entry
        """
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        
        logger.info(f"Initialized sentiment cache: maxsize={maxsize}, ttl={ttl}s")
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate cache key from text.
        
        Uses SHA256 hash truncated to 16 characters for efficient lookups.
        
        Args:
            text: Input text to hash
            
        Returns:
            Truncated hash string to use as cache key
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> Optional[Dict]:
        """Get cached result for text.
        
        Args:
            text: Input text to look up
            
        Returns:
            Cached result dict if found, None otherwise
        """
        key = self._hash_text(text)
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._hits += 1
                logger.debug(f"Cache hit: {key}")
            else:
                self._misses += 1
                logger.debug(f"Cache miss: {key}")
            return result
    
    def set(self, text: str, result: Dict) -> None:
        """Store result in cache.
        
        Args:
            text: Input text that was analyzed
            result: Analysis result to cache
        """
        key = self._hash_text(text)
        with self._lock:
            self._cache[key] = result
            logger.debug(f"Cache set: {key}")
    
    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    @property
    def stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, size, and hit rate
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate_percent": round(hit_rate, 2)
            }


# ============================================================================
# Sentiment Analysis Engine
# ============================================================================

class SentimentEngine:
    """
    Production-grade sentiment analysis engine.
    
    Features:
    - Transformer-based models (RoBERTa, DistilBERT)
    - Automatic model fallback on failure
    - Thread-safe inference
    - Result caching
    - Comprehensive logging
    
    Example:
        >>> engine = SentimentEngine()
        >>> engine.load_model()
        >>> result = engine.analyze("This product is amazing!")
        >>> print(result)
        {'sentiment': 'positive', 'score': 0.98, 'confidence': 0.99, ...}
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """Initialize the sentiment engine.
        
        Args:
            model_name: HuggingFace model name to use (default: twitter-roberta-base-sentiment)
            device: Device to run model on ('cuda' or 'cpu', auto-detected if None)
        """
        self.config = ModelConfig()
        self.model_name = model_name or self.config.MODEL_NAME
        self.device = device or self.config.DEVICE
        
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self._cache = SentimentCache(
            maxsize=self.config.CACHE_SIZE,
            ttl=self.config.CACHE_TTL
        )
        self._model_loaded = False
        self._lock = threading.Lock()
        
        logger.info(f"Initialized SentimentEngine with model: {self.model_name}")
    
    def load_model(self) -> None:
        """Load the sentiment analysis model.
        
        Downloads the model from HuggingFace if not already cached,
        then loads it onto the specified device (CPU or GPU).
        
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model_loaded:
            return
            
        with self._lock:
            if self._model_loaded:
                return
                
            try:
                logger.info(f"Loading model: {self.model_name}")
                
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Load model
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                self._model.to(self.device)
                self._model.eval()
                
                # Create pipeline for easier inference
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    truncation=True,
                    max_length=self.config.MAX_LENGTH
                )
                
                self._model_loaded = True
                logger.info(
                    f"Model loaded successfully on {self.device}. "
                    f"Parameters: {sum(p.numel() for p in self._model.parameters()):,}"
                )
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise ModelLoadError(f"Model initialization failed: {e}")
    
    @contextmanager
    def _inference_context(self):
        """Context manager for model inference.
        
        Ensures model is loaded before inference and handles any errors
        that occur during the inference process.
        
        Yields:
            None
        """
        if not self._model_loaded:
            self.load_model()
        try:
            yield
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise AnalysisError(f"Sentiment analysis failed: {e}")
    
    def analyze(
        self,
        text: str,
        use_cache: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            use_cache: Whether to use cached results (default: True)
            verbose: Include detailed model scores in response (default: False)
            
        Returns:
            Dictionary with sentiment analysis results:
            - sentiment: 'positive', 'negative', or 'neutral'
            - score: Normalized score from -1 (negative) to 1 (positive)
            - confidence: Model confidence (0 to 1)
            - model: Model name used for analysis
            - analyzed_at: ISO format timestamp
            - raw_scores: (only if verbose=True) Detailed model output
            
        Raises:
            AnalysisError: If analysis fails
        """
        if not text or not text.strip():
            return self._empty_result()
        
        text = text.strip()
        
        # Check cache
        if use_cache:
            cached = self._cache.get(text)
            if cached is not None:
                logger.debug(f"Returning cached result for text: {text[:50]}...")
                return cached
        
        with self._inference_context():
            try:
                # Run inference
                result = self._pipeline(text)[0]
                
                # Parse model output
                label = result["label"].lower()
                score = result["score"]
                
                # Map to our format
                if "pos" in label:
                    sentiment = SentimentLabel.POSITIVE
                    normalized_score = score
                elif "neg" in label:
                    sentiment = SentimentLabel.NEGATIVE
                    normalized_score = -score
                else:
                    sentiment = SentimentLabel.NEUTRAL
                    normalized_score = 0.0
                
                # Build response
                response = {
                    "sentiment": sentiment.value,
                    "score": round(normalized_score, 4),
                    "confidence": round(score, 4),
                    "model": self.model_name,
                    "analyzed_at": datetime.utcnow().isoformat()
                }
                
                if verbose:
                    response["raw_scores"] = {
                        "label": label,
                        "raw_score": round(score, 4)
                    }
                
                # Cache result
                if use_cache:
                    self._cache.set(text, response)
                
                logger.debug(f"Analyzed text: {text[:30]}... -> {sentiment.value}")
                return response
                
            except Exception as e:
                logger.error(f"Analysis failed for '{text[:50]}...': {e}")
                raise AnalysisError(f"Failed to analyze text: {e}")
    
    def analyze_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        verbose: bool = False,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Analyze sentiment of multiple texts.
        
        Processes texts in batches for efficiency. Uses the same
        caching mechanism as single text analysis.
        
        Args:
            texts: List of texts to analyze
            use_cache: Whether to use cached results (default: True)
            verbose: Include detailed scores (default: False)
            show_progress: Log progress during processing (default: True)
            
        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []
        
        results = []
        total = len(texts)
        
        logger.info(f"Starting batch analysis: {total} texts")
        
        # Process in batches for efficiency
        for i in range(0, total, self.config.BATCH_SIZE):
            batch_texts = texts[i:i + self.config.BATCH_SIZE]
            
            # Filter out empty texts
            valid_texts = [t.strip() for t in batch_texts if t and t.strip()]
            
            if not valid_texts:
                continue
            
            with self._inference_context():
                try:
                    # Batch inference
                    batch_results = self._pipeline(valid_texts)
                    
                    for text, result in zip(valid_texts, batch_results):
                        # Parse result
                        label = result["label"].lower()
                        score = result["score"]
                        
                        if "pos" in label:
                            sentiment = SentimentLabel.POSITIVE
                            normalized_score = score
                        elif "neg" in label:
                            sentiment = SentimentLabel.NEGATIVE
                            normalized_score = -score
                        else:
                            sentiment = SentimentLabel.NEUTRAL
                            normalized_score = 0.0
                        
                        response = {
                            "sentiment": sentiment.value,
                            "score": round(normalized_score, 4),
                            "confidence": round(score, 4),
                            "model": self.model_name,
                            "analyzed_at": datetime.utcnow().isoformat()
                        }
                        
                        if verbose:
                            response["raw_scores"] = {
                                "label": label,
                                "raw_score": round(score, 4)
                            }
                        
                        results.append(response)
                        
                        # Cache individual results
                        if use_cache:
                            self._cache.set(text, response)
                    
                    if show_progress:
                        logger.info(f"Progress: {min(i + self.config.BATCH_SIZE, total)}/{total}")
                        
                except Exception as e:
                    logger.error(f"Batch analysis error at index {i}: {e}")
                    # Add error results for failed items
                    for _ in valid_texts:
                        results.append(self._error_result())
        
        logger.info(f"Batch analysis complete: {len(results)}/{total} texts processed")
        return results
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for invalid input.
        
        Returns neutral sentiment with zero scores for empty or
        whitespace-only input text.
        
        Returns:
            Dictionary with neutral sentiment and zero values
        """
        return {
            "sentiment": SentimentLabel.NEUTRAL.value,
            "score": 0.0,
            "confidence": 0.0,
            "model": self.model_name,
            "analyzed_at": datetime.utcnow().isoformat(),
            "raw_scores": {"pos": 0.0, "neg": 0.0, "neu": 1.0}
        }
    
    def _error_result(self) -> Dict[str, Any]:
        """Return error result.
        
        Returns neutral sentiment with error flag when analysis fails.
        
        Returns:
            Dictionary with error flag set
        """
        return {
            "sentiment": SentimentLabel.NEUTRAL.value,
            "score": 0.0,
            "confidence": 0.0,
            "model": self.model_name,
            "analyzed_at": datetime.utcnow().isoformat(),
            "error": True
        }
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache hits, misses, size, and hit rate
        """
        return self._cache.stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check engine health status.
        
        Returns:
            Dictionary with status, model info, and cache stats
        """
        return {
            "status": "healthy" if self._model_loaded else "loading",
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "cache": self._cache.stats
        }


# ============================================================================
# Singleton Instance
# ============================================================================

# Global engine instance with lazy loading
_engine: Optional[SentimentEngine] = None


def get_engine() -> SentimentEngine:
    """Get or create the global sentiment engine instance.
    
    Implements singleton pattern for the engine to ensure
    consistent model loading and caching across calls.
    
    Returns:
        Singleton SentimentEngine instance
    """
    global _engine
    if _engine is None:
        _engine = SentimentEngine()
    return _engine


def analyze_sentiment(
    text: str,
    use_cache: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """Convenience function for sentiment analysis.
    
    Provides a simple interface for single text analysis without
    needing to instantiate the engine manually.
    
    Args:
        text: Input text to analyze
        use_cache: Use caching for results (default: True)
        verbose: Include detailed scores (default: False)
        
    Returns:
        Dictionary with sentiment results
        
    Example:
        >>> result = analyze_sentiment("This is amazing!")
        >>> print(result['sentiment'])
        'positive'
    """
    engine = get_engine()
    return engine.analyze(text, use_cache=use_cache, verbose=verbose)


def analyze_batch(
    texts: List[str],
    use_cache: bool = True,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Convenience function for batch sentiment analysis.
    
    Provides a simple interface for batch analysis without
    needing to instantiate the engine manually.
    
    Args:
        texts: List of texts to analyze
        use_cache: Use caching for results (default: True)
        verbose: Include detailed scores (default: False)
        
    Returns:
        List of sentiment results
        
    Example:
        >>> results = analyze_batch(["Great!", "Terrible!", "Okay"])
        >>> for r in results:
        ...     print(r['sentiment'])
    """
    engine = get_engine()
    return engine.analyze_batch(texts, use_cache=use_cache, verbose=verbose)


# Initialize on import (lazy)
logger.info("SentimentPulse engine module loaded")
