# 🔮 SentimentPulse

Production-grade sentiment analysis API & CLI powered by state-of-the-art transformer models.

[![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python)](https://www.python.org/)
[![API](https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/-License-MIT-orange?style=flat)](LICENSE)

*Sentiment analysis for developers who need accuracy, scalability, and reliability.*

---

## What is SentimentPulse?

SentimentPulse is a **production-ready** sentiment analysis system built for developers who need **accurate, scalable, and reliable** text sentiment detection. Whether you're analyzing customer feedback, monitoring social media, or building AI-powered products — SentimentPulse delivers enterprise-grade results.

## Why Choose SentimentPulse?

| Traditional Tools | SentimentPulse |
|-------------------|----------------|
| Basic lexicon-based (VADER, TextBlob) | **Transformer-based** (RoBERTa, DistilBERT) |
| Limited accuracy | **92% accuracy** on benchmark datasets |
| No caching | **Intelligent caching** for speed |
| No rate limiting | **Built-in rate limiting** |
| Basic error handling | **Production error handling** with Sentry |

---

## Features

### Core Capabilities
- 🔮 **Transformer-Powered** - Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` for state-of-the-art accuracy
- ⚡ **High Performance** - Batch processing with GPU acceleration support
- 💾 **Smart Caching** - TTLCache for optimal memory usage and speed
- 🛡️ **Rate Limiting** - Built-in protection against abuse (60 req/min single, 20 req/min batch)
- 🔄 **Auto Fallback** - Multiple model fallback for resilience
- 📊 **Batch Processing** - Process up to 500 texts in a single request

### Developer Experience
- 📡 **RESTful API** - FastAPI with OpenAPI/Swagger documentation
- 💻 **Beautiful CLI** - Rich-formatted terminal interface with interactive mode
- 🛡️ **Type Safety** - Full PyDantic validation
- 📈 **Monitoring** - Sentry integration for error tracking
- 🐳 **Docker Ready** - Production Dockerfile included
- ⚙️ **Environment Config** - PyYAML Settings for configuration

---

## Tech Stack

<div align="center">

| Category | Technology |
|----------|------------|
| **ML/AI** | ![Transformers](https://img.shields.io/badge/-Transformers-FF6B6B?logo=huggingface) ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch) |
| **API** | ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi) ![Uvicorn](https://img.shields.io/badge/-Uvicorn-444444?logo=) |
| **CLI** | ![Click](https://img.shields.io/badge/-Click-007ACC?logo=click) ![Rich](https://img.shields.io/badge/-Rich-FF5555?logo=) |
| **Caching** | ![Cachetools](https://img.shields.io/badge/-Cachetools-843953?logo=) |
| **Testing** | ![Pytest](https://img.shields.io/badge/-Pytest-0A9EDC?logo=pytest) |

</div>

---

## Installation

### Prerequisites

```bash
Python 3.10+
pip (package manager)
```

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Himal-Badu/SentimentPulse.git
cd SentimentPulse

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Development Install

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Verify installation
sentiment --version
```

---

## Usage

### CLI Usage

#### Single Text Analysis

```bash
# Quick analyze
sentiment analyze "I absolutely love this product! It's amazing!"

# With detailed output
sentiment analyze "This is terrible service" --verbose

# Output as JSON
sentiment analyze "Great job team!" --json
```

#### Interactive Shell

```bash
# Start interactive mode
sentiment shell

# Then type your text...
```

#### Batch Processing

```bash
# Analyze texts from file (one per line)
sentiment batch reviews.txt

# Analyze JSON file
sentiment batch data.json --output results.json

# Limit number of texts
sentiment batch large_file.txt --limit 100
```

### Python API Usage

```python
from sentimentpulse import analyze_sentiment, analyze_batch

# Single text analysis
result = analyze_sentiment("This product is incredible!")
print(result)
# {
#     "sentiment": "positive",
#     "score": 0.9452,
#     "confidence": 0.98,
#     "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
#     "analyzed_at": "2026-03-23T12:00:00Z"
# }

# Batch analysis
texts = [
    "Love this! 🙏",
    "Worst experience ever 😡",
    "It's okay, not great 🤔"
]
results = analyze_batch(texts)
```

---

## API Reference

### Start the API Server

```bash
# Development server
uvicorn api.main:app --reload

# Production server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Endpoints

#### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "cache_stats": {
    "hits": 150,
    "misses": 50,
    "size": 200,
    "hit_rate_percent": 75.0
  }
}
```

#### Analyze Single Text

```
POST /api/v1/analyze
```

**Request:**
```json
{
  "text": "I absolutely love this new feature!",
  "verbose": false,
  "use_cache": true
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "score": 0.9452,
  "confidence": 0.9823,
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "analyzed_at": "2026-03-23T12:00:00Z"
}
```

#### Batch Analysis

```
POST /api/v1/analyze/batch
```

**Request:**
```json
{
  "texts": [
    "Great product!",
    "Terrible service",
    "Average quality"
  ],
  "verbose": false,
  "use_cache": true
}
```

**Response:**
```json
{
  "results": [
    {
      "sentiment": "positive",
      "score": 0.9452,
      "confidence": 0.9823,
      "model": "...",
      "analyzed_at": "2026-03-23T12:00:00Z"
    },
    {
      "sentiment": "negative",
      "score": -0.9234,
      "confidence": 0.9678,
      "model": "...",
      "analyzed_at": "2026-03-23T12:00:00Z"
    },
    {
      "sentiment": "neutral",
      "score": 0.0,
      "confidence": 0.89,
      "model": "...",
      "analyzed_at": "2026-03-23T12:00:00Z"
    }
  ],
  "total": 3,
  "processed_at": "2026-03-23T12:00:00Z",
  "processing_time_ms": 125.5
}
```

#### Cache Management

```
GET /api/v1/cache/stats
DELETE /api/v1/cache
```

---

## Deployment

### Docker Deployment

```bash
# Build the image
docker build -t sentimentpulse:latest .

# Run the container
docker run -d -p 8000:8000 --name sentimentpulse sentimentpulse:latest

# Check logs
docker logs -f sentimentpulse

# Stop container
docker stop sentimentpulse
```

### Docker Compose (Recommended)

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - SENTRY_DSN=your_sentry_dsn_here
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

See [DEPLOY.md](DEPLOY.md) for detailed production deployment instructions including:

- Nginx reverse proxy setup
- System service configuration
- Environment variables
- Monitoring & alerting

---

## Future Scope

- [ ] **Multi-language Support** - Add models for 10+ languages
- [ ] **Emotion Detection** - Beyond sentiment: joy, anger, sadness, etc.
- [ ] **Aspect-Based Analysis** - Sentiment per feature/aspect
- [ ] **Real-time Streaming** - WebSocket support for live analytics
- [ ] **Model Fine-tuning** - Custom model training for domain-specific data
- [ ] **Redis Caching** - Distributed cache for multi-instance deployments
- [ ] **GraphQL API** - Alternative API for complex queries

---

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

<div align="center">

**Built by Himal Badu, AI Founder**

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/himal-badu)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github)](https://github.com/Himal-Badu)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat&logo=gmail)](mailto:himalbaduhimalbadu@gmail.com)

*Building the future of AI, one commit at a time.*

</div>

---

<div align="center">

[![Star](https://img.shields.io/badge/-Star-FFD700?style=flat&logo=)](https://github.com/Himal-Badu/SentimentPulse/stargazers)
[![Sponsor](https://img.shields.io/badge/-Sponsor-FF6B6B?style=flat&)](https://github.com/sponsors/Himal-Badu)

If you find SentimentPulse useful, please consider giving it a star!

</div>
