"""
SentimentPulse - Configuration management utilities
Built by Himal Badu, AI Founder

Utilities for managing configuration and environment variables.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

import yaml


@dataclass
class ConfigSource:
    """Configuration source with priority."""
    source: str  # env, file, default
    key: str
    value: Any


class ConfigManager:
    """Manages configuration from multiple sources."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config: Dict[str, Any] = {}
        self._sources: Dict[str, ConfigSource] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from all sources."""
        # Load from file if exists
        if self.config_file and os.path.exists(self.config_file):
            self._load_from_file()
        
        # Load from environment
        self._load_from_env()
    
    def _load_from_file(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, "r") as f:
                data = yaml.safe_load(f) or {}
                for key, value in data.items():
                    self._config[key] = value
                    self._sources[key] = ConfigSource("file", key, value)
        except Exception:
            pass
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Define known environment variables
        env_vars = [
            "SENTIMENT_MODEL",
            "BATCH_SIZE",
            "MAX_LENGTH",
            "DEVICE",
            "CACHE_ENABLED",
            "CACHE_SIZE",
            "CACHE_TTL",
            "REDIS_URL",
            "RATE_LIMIT_ENABLED",
            "RATE_LIMIT_PER_MINUTE",
            "LOG_LEVEL",
            "LOG_DIR",
            "API_KEY",
            "SECRET_KEY",
            "SENTRY_DSN",
            "ENVIRONMENT",
            "HOST",
            "PORT",
            "WORKERS",
            "RELOAD",
            "DEBUG",
        ]
        
        for var in env_vars:
            value = os.getenv(var)
            if value is not None:
                # Try to parse as int/bool
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)
                
                self._config[var.lower()] = value
                self._sources[var.lower()] = ConfigSource("env", var, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key.lower(), default)
    
    def set(self, key: str, value: Any, source: str = "runtime"):
        """Set configuration value."""
        self._config[key.lower()] = value
        self._sources[key.lower()] = ConfigSource(source, key, value)
    
    def get_source(self, key: str) -> Optional[ConfigSource]:
        """Get the source of a configuration value."""
        return self._sources.get(key.lower())
    
    def all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "config": self._config,
            "sources": {
                k: {"source": v.source, "key": v.key}
                for k, v in self._sources.items()
            }
        }


# Global config manager
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def validate_config() -> Dict[str, Any]:
    """Validate configuration and return warnings."""
    warnings = []
    config = get_config_manager()
    
    # Check for required settings
    if not config.get("sentiment_model"):
        warnings.append("SENTIMENT_MODEL not set, using default")
    
    # Check for potential issues
    if config.get("cache_size", 0) < 100:
        warnings.append("Cache size is very small, consider increasing")
    
    if config.get("batch_size", 0) > 100:
        warnings.append("Large batch size may cause memory issues")
    
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "config": config.all()
    }
