# Development Guide

This guide covers development setup, coding standards, and contribution guidelines for SentimentPulse.

*Built by Himal Badu, AI Founder*

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip or poetry
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Himal-Badu/SentimentPulse.git
cd SentimentPulse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Start development server
python -m api.main
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Model Configuration
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
BATCH_SIZE=32
MAX_LENGTH=512
DEVICE=cpu  # cpu or cuda

# Cache Configuration
CACHE_ENABLED=true
CACHE_SIZE=1000
CACHE_TTL=3600

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Security
API_KEY=your-api-key
SECRET_KEY=your-secret-key

# Monitoring
SENTRY_DSN=your-sentry-dsn
ENVIRONMENT=development
```

## Coding Standards

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Use docstrings for all public functions
- Maximum line length: 100 characters

### Naming Conventions

- Classes: `CamelCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Example

```python
from typing import Dict, List, Optional


class SentimentAnalyzer:
    """Analyzes sentiment of text.
    
    Args:
        model_name: Name of the model to use
        
    Attributes:
        _model: The loaded model
    """
    
    def __init__(self, model_name: str = "default") -> None:
        self._model_name = model_name
        self._model = None
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment results
        """
        if not text:
            return {"sentiment": "neutral", "score": 0.0}
        
        # Implementation here
        return {"sentiment": "positive", "score": 0.95}
```

## Project Structure

```
SentimentPulse/
├── api/                    # API modules
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic models
│   ├── websocket.py       # WebSocket endpoints
│   ├── analytics.py       # Analytics endpoints
│   └── errors.py          # Error handling
├── cli/                    # CLI modules
│   ├── main.py            # CLI entry point
│   ├── shell.py           # Interactive shell
│   └── models.py          # Model management
├── sentimentpulse/         # Core library
│   ├── engine.py          # Analysis engine
│   ├── config.py          # Configuration
│   ├── utils.py           # Utilities
│   ├── monitoring.py      # Health monitoring
│   ├── storage.py         # Storage layer
│   └── rate_limit.py     # Rate limiting
├── tests/                  # Test suite
├── examples.py            # Usage examples
├── setup.py               # Package setup
└── pyproject.toml        # Project config
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sentimentpulse --cov-report=html

# Run specific test file
pytest tests/test_engine.py

# Run specific test
pytest tests/test_engine.py::TestEngine::test_analyze
```

### Writing Tests

```python
import pytest
from sentimentpulse import analyze_sentiment


class TestSentimentAnalysis:
    """Tests for sentiment analysis."""
    
    def test_positive_sentiment(self):
        """Test positive text returns positive sentiment."""
        result = analyze_sentiment("I love this!")
        assert result["sentiment"] == "positive"
    
    def test_negative_sentiment(self):
        """Test negative text returns negative sentiment."""
        result = analyze_sentiment("This is terrible!")
        assert result["sentiment"] == "negative"
    
    def test_empty_text(self):
        """Test empty text returns neutral."""
        result = analyze_sentiment("")
        assert result["sentiment"] == "neutral"
```

## Building

### Package Installation

```bash
# Install in development mode
pip install -e .

# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

### Docker

```bash
# Build image
docker build -t sentimentpulse .

# Run container
docker run -p 8000:8000 sentimentpulse

# Run with docker-compose
docker-compose up -d
```

## Contributing

### Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests
5. Submit a pull request

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add batch analysis endpoint
fix: Resolve cache eviction issue
docs: Update API documentation
test: Add tests for new features
refactor: Improve error handling
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release commit
4. Tag release
5. Build and upload package

---

*Built by Himal Badu, AI Founder*
