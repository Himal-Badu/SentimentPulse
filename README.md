# 🔮 SentimentPulse

Production-grade sentiment analysis API & CLI powered by state-of-the-art transformer models.

[![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python)](https://www.python.org/)
[![API](https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/-License-MIT-orange?style=flat)](LICENSE)

*Real-time sentiment analysis for developers who need accuracy, scalability, and reliability.*

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
| Basic error handling | **Production error handling** |

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

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **ML/AI** | Transformers, PyTorch |
| **API** | FastAPI, Uvicorn |
| **CLI** | Click, Rich |
| **Caching** | Cachetools |
| **Testing** | Pytest |

---

## Installation

```bash
git clone https://github.com/Himal-Badu/SentimentPulse.git
cd SentimentPulse
pip install -r requirements.txt
```

---

## Usage

### CLI
```bash
sentiment analyze "I love this product!"
sentiment batch reviews.txt
sentiment shell
```

### API
```bash
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

---

## Deployment

### Docker
```bash
docker build -t sentimentpulse:latest .
docker run -d -p 8000:8000 sentimentpulse:latest
```

---

## Future Scope

- Multi-language Support
- Emotion Detection
- Aspect-Based Analysis
- Real-time Streaming (WebSocket)
- Model Fine-tuning

---

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) first.

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Built by Himal Badu, AI Founder**

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/himal-badu)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github)](https://github.com/Himal-Badu)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat&logo=gmail)](mailto:himalbaduhimalbadu@gmail.com)

*Building the future of AI, one commit at a time.*

## 📊 Project Stats

![GitHub Stars](https://img.shields.io/github/stars/Himal-Badu/SentimentPulse)
![GitHub Forks](https://img.shields.io/github/forks/Himal-Badu/SentimentPulse)
![GitHub Issues](https://img.shields.io/github/issues/Himal-Badu/SentimentPulse)
![GitHub License](https://img.shields.io/github/license/Himal-Badu/SentimentPulse)
![Python](https://img.shields.io/badge/Python-3.9+-blue)

## 🔗 Links

- [Documentation](https://github.com/Himal-Badu/SentimentPulse#readme)
- [API Reference](https://github.com/Himal-Badu/SentimentPulse/blob/master/api/main.py)
- [Contributing](https://github.com/Himal-Badu/SentimentPulse/blob/master/CONTRIBUTING.md)
