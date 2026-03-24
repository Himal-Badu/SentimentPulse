"""
Configuration module for SentimentPulse
Built by Himal Badu, AI Founder

Centralized configuration management with environment variable support.
"""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from pydantic_settings import BaseSettings
from pydantic import Field


@dataclass
class APIConfig:
    """API server configuration."""
    title: str = "SentimentPulse API"
    version: str = "2.0.0"
    description: str = "Production-grade sentiment analysis API"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    debug: bool = False


@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_length: int = 512
    batch_size: int = 32
    fallback_models: List[str] = field(default_factory=lambda: [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "nlptown/bert-base-multilingual-uncased-sentiment"
    ])
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    max_size: int = 1000
    ttl: int = 3600  # seconds
    redis_url: Optional[str] = None


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    batch_requests_per_minute: int = 20


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs"
    format_string: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    rotation: str = "1 day"
    retention: str = "30 days"


@dataclass
class SecurityConfig:
    """Security configuration."""
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_hosts: List[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None
    secret_key: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    sentry_dsn: Optional[str] = None
    enable_metrics: bool = True
    health_check_interval: int = 60


class Settings(BaseSettings):
    """Main settings class combining all configurations."""
    
    # API
    api_title: str = Field(default="SentimentPulse API", env="API_TITLE")
    api_version: str = Field(default="2.0.0", env="API_VERSION")
    api_description: str = Field(
        default="Production-grade sentiment analysis API",
        env="API_DESCRIPTION"
    )
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Model
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        env="SENTIMENT_MODEL"
    )
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_length: int = Field(default=512, env="MAX_LENGTH")
    device: str = Field(default="auto", env="DEVICE")
    
    # Cache
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_size: int = Field(default=1000, env="CACHE_SIZE")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    batch_rate_limit_per_minute: int = Field(default=20, env="BATCH_RATE_LIMIT_PER_MINUTE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = Field(default="logs", env="LOG_DIR")
    
    # Security
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    secret_key: Optional[str] = Field(default=None, env="SECRET_KEY")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    def get_device(self) -> str:
        """Get device to use for inference."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
