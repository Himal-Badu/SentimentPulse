"""
SentimentPulse - Health monitoring and metrics
Built by Himal Badu, AI Founder
"""

import time
import psutil
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_used_percent: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class APIMetrics:
    """API performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    requests_per_minute: float = 0.0


class HealthMonitor:
    """Monitor system health and API metrics."""
    
    def __init__(self):
        self._start_time = time.time()
        self._api_metrics = APIMetrics()
        self._request_times: list = []
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_used_percent=disk.percent
        )
    
    def record_request(self, response_time_ms: float, success: bool = True):
        """Record API request metrics."""
        self._api_metrics.total_requests += 1
        
        if success:
            self._api_metrics.successful_requests += 1
        else:
            self._api_metrics.failed_requests += 1
        
        # Track response times (keep last 100)
        self._request_times.append(response_time_ms)
        if len(self._request_times) > 100:
            self._request_times.pop(0)
        
        # Calculate average
        if self._request_times:
            self._api_metrics.average_response_time_ms = sum(self._request_times) / len(self._request_times)
        
        # Calculate uptime
        self._api_metrics.uptime_seconds = time.time() - self._start_time
    
    def get_api_metrics(self) -> Dict[str, Any]:
        """Get API metrics."""
        uptime = time.time() - self._start_time
        if uptime > 0:
            self._api_metrics.requests_per_minute = (self._api_metrics.total_requests / uptime) * 60
        
        return {
            "total_requests": self._api_metrics.total_requests,
            "successful_requests": self._api_metrics.successful_requests,
            "failed_requests": self._api_metrics.failed_requests,
            "average_response_time_ms": round(self._api_metrics.average_response_time_ms, 2),
            "uptime_seconds": round(uptime, 2),
            "requests_per_minute": round(self._api_metrics.requests_per_minute, 2)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        system = self.get_system_metrics()
        
        # Determine health based on resource usage
        status = "healthy"
        if system.memory_percent > 90 or system.disk_used_percent > 90:
            status = "critical"
        elif system.memory_percent > 75 or system.disk_used_percent > 80:
            status = "degraded"
        
        return {
            "status": status,
            "system": {
                "cpu_percent": round(system.cpu_percent, 2),
                "memory_percent": round(system.memory_percent, 2),
                "memory_used_mb": round(system.memory_used_mb, 2),
                "disk_used_percent": round(system.disk_used_percent, 2)
            },
            "api": self.get_api_metrics(),
            "timestamp": system.timestamp
        }


# Global health monitor
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
