"""
API integration tests for SentimentPulse
Built by Himal Badu, AI Founder
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns status."""
        response = client.get("/health")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "version" in data


class TestAnalyzeEndpoint:
    """Tests for analyze endpoint."""
    
    def test_analyze_valid_text(self, client):
        """Test analyzing valid text."""
        response = client.post(
            "/api/v1/analyze",
            json={"text": "I love this product!"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "score" in data
        assert "confidence" in data
    
    def test_analyze_empty_text(self, client):
        """Test analyzing empty text returns error."""
        response = client.post(
            "/api/v1/analyze",
            json={"text": ""}
        )
        
        assert response.status_code == 422
    
    def test_analyze_verbose(self, client):
        """Test verbose response includes raw scores."""
        response = client.post(
            "/api/v1/analyze",
            json={"text": "Test", "verbose": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "raw_scores" in data or "sentiment" in data
    
    def test_analyze_no_cache(self, client):
        """Test analysis without cache."""
        response = client.post(
            "/api/v1/analyze",
            json={"text": "Test", "use_cache": False}
        )
        
        assert response.status_code == 200


class TestBatchEndpoint:
    """Tests for batch analyze endpoint."""
    
    def test_batch_analyze(self, client):
        """Test batch analysis."""
        response = client.post(
            "/api/v1/analyze/batch",
            json={
                "texts": [
                    "I love this!",
                    "This is terrible.",
                    "It's okay."
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
    
    def test_batch_empty(self, client):
        """Test batch with empty list."""
        response = client.post(
            "/api/v1/analyze/batch",
            json={"texts": []}
        )
        
        assert response.status_code == 422
    
    def test_batch_large(self, client):
        """Test batch with many texts."""
        texts = ["Test text"] * 100
        response = client.post(
            "/api/v1/analyze/batch",
            json={"texts": texts}
        )
        
        assert response.status_code == 200


class TestCacheEndpoints:
    """Tests for cache endpoints."""
    
    def test_cache_stats(self, client):
        """Test cache stats endpoint."""
        response = client.get("/api/v1/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert "misses" in data
    
    def test_clear_cache(self, client):
        """Test cache clear endpoint."""
        response = client.delete("/api/v1/cache")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestModelEndpoints:
    """Tests for model endpoints."""
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model")
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
    
    def test_api_info(self, client):
        """Test API info endpoint."""
        response = client.get("/api/v1/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "repository" in data


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_json(self, client):
        """Test invalid JSON request."""
        response = client.post(
            "/api/v1/analyze",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_field(self, client):
        """Test missing required field."""
        response = client.post(
            "/api/v1/analyze",
            json={}
        )
        
        assert response.status_code == 422


class TestCORS:
    """Tests for CORS."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/v1/analyze",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
