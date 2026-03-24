"""
SentimentPulse - Rate limiting and throttling
Built by Himal Badu, AI Founder
"""

import time
import threading
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    
    def get_key(self, identifier: str) -> str:
        """Generate cache key for rate limit."""
        return hashlib.md5(f"{identifier}:{self.requests_per_minute}".encode()).hexdigest()


class RateLimiter:
    """Token bucket rate limiter with multiple time windows."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": self.config.burst_limit,
            "last_update": time.time(),
            "minute_count": 0,
            "minute_reset": time.time(),
            "hour_count": 0,
            "hour_reset": time.time(),
        })
        self._lock = threading.RLock()
    
    def check_limit(self, identifier: str) -> tuple[bool, Dict]:
        """Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (IP, API key, user ID)
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        with self._lock:
            bucket = self._buckets[identifier]
            current_time = time.time()
            
            # Reset counters if needed
            if current_time - bucket["minute_reset"] > 60:
                bucket["minute_count"] = 0
                bucket["minute_reset"] = current_time
            
            if current_time - bucket["hour_reset"] > 3600:
                bucket["hour_count"] = 0
                bucket["hour_reset"] = current_time
            
            # Check burst limit (token bucket)
            time_passed = current_time - bucket["last_update"]
            bucket["tokens"] = min(
                self.config.burst_limit,
                bucket["tokens"] + time_passed * (self.config.requests_per_minute / 60)
            )
            bucket["last_update"] = current_time
            
            if bucket["tokens"] < 1:
                return False, self._get_limit_info(identifier, bucket)
            
            # Check minute limit
            if bucket["minute_count"] >= self.config.requests_per_minute:
                return False, self._get_limit_info(identifier, bucket)
            
            # Check hour limit
            if bucket["hour_count"] >= self.config.requests_per_hour:
                return False, self._get_limit_info(identifier, bucket)
            
            # Consume token
            bucket["tokens"] -= 1
            bucket["minute_count"] += 1
            bucket["hour_count"] += 1
            
            return True, self._get_limit_info(identifier, bucket)
    
    def _get_limit_info(self, identifier: str, bucket: Dict) -> Dict:
        """Get rate limit information."""
        return {
            "identifier": identifier,
            "remaining_tokens": int(bucket["tokens"]),
            "requests_this_minute": bucket["minute_count"],
            "requests_this_hour": bucket["hour_count"],
            "limit_per_minute": self.config.requests_per_minute,
            "limit_per_hour": self.config.requests_per_hour,
            "reset_at": datetime.fromtimestamp(bucket["minute_reset"] + 60).isoformat()
        }
    
    def reset(self, identifier: str) -> None:
        """Reset rate limit for identifier."""
        with self._lock:
            if identifier in self._buckets:
                del self._buckets[identifier]
    
    def get_status(self, identifier: str) -> Dict:
        """Get current rate limit status."""
        with self._lock:
            bucket = self._buckets.get(identifier, {})
            if not bucket:
                return {
                    "remaining_tokens": self.config.burst_limit,
                    "requests_this_minute": 0,
                    "requests_this_hour": 0
                }
            return self._get_limit_info(identifier, bucket)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more accurate rate limiting."""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.RLock()
    
    def is_allowed(self, identifier: str) -> tuple[bool, Dict]:
        """Check if request is allowed using sliding window."""
        with self._lock:
            current_time = time.time()
            window_start = current_time - self.window_seconds
            
            # Remove old requests outside window
            self._requests[identifier] = [
                ts for ts in self._requests[identifier]
                if ts > window_start
            ]
            
            # Check limit
            if len(self._requests[identifier]) >= self.max_requests:
                oldest = min(self._requests[identifier])
                retry_after = int(oldest + self.window_seconds - current_time) + 1
                
                return False, {
                    "allowed": False,
                    "remaining": 0,
                    "limit": self.max_requests,
                    "window_seconds": self.window_seconds,
                    "retry_after_seconds": retry_after
                }
            
            # Add current request
            self._requests[identifier].append(current_time)
            
            return True, {
                "allowed": True,
                "remaining": self.max_requests - len(self._requests[identifier]),
                "limit": self.max_requests,
                "window_seconds": self.window_seconds
            }
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        with self._lock:
            current_time = time.time()
            window_start = current_time - self.window_seconds
            
            recent_requests = [
                ts for ts in self._requests[identifier]
                if ts > window_start
            ]
            
            return max(0, self.max_requests - len(recent_requests))


# Global rate limiter instances
_rate_limiters: Dict[str, SlidingWindowRateLimiter] = {}


def get_rate_limiter(limit: int = 60, window: int = 60) -> SlidingWindowRateLimiter:
    """Get or create a rate limiter."""
    key = f"{limit}:{window}"
    if key not in _rate_limiters:
        _rate_limiters[key] = SlidingWindowRateLimiter(limit, window)
    return _rate_limiters[key]
