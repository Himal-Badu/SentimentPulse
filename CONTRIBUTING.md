# Contributing to SentimentPulse

Thank you for your interest in contributing to SentimentPulse!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Himal-Badu/SentimentPulse.git
cd SentimentPulse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Project Structure

```
SentimentPulse/
├── api/                 # FastAPI application
│   ├── main.py         # API endpoints
│   └── models.py      # Request/response schemas
├── cli/               # CLI application
│   └── main.py        # CLI commands
├── sentimentpulse/    # Core library
│   ├── engine.py      # Sentiment analysis engine
│   ├── config.py      # Configuration
│   └── utils.py       # Utilities
├── tests/             # Test suite
│   ├── test_analyzer.py
│   ├── test_cli.py
│   └── test_extended.py
└── docs/              # Documentation
```

## Coding Standards

- Use type hints for all function parameters and return values
- Add docstrings to all public functions and classes
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Keep functions focused and small (single responsibility)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sentimentpulse --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py

# Run tests matching pattern
pytest -k "test_positive"
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass
5. Update documentation if needed
6. Commit with descriptive messages
7. Push to your fork
8. Open a Pull Request

## Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add: New feature description`
- `Fix: Bug fix description`
- `Refactor: Code improvement`
- `Docs: Documentation update`
- `Test: Test coverage addition`

## Code Style

```python
# Good
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with sentiment results
    """
    pass

# Avoid
def analyze(text):  # No type hints
    pass
```

## Reporting Issues

When reporting issues, include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

## License

By contributing to SentimentPulse, you agree that your contributions will be licensed under the MIT License.
