# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-03-24

### Added
- Production-grade sentiment analysis engine with transformer models
- RESTful FastAPI with OpenAPI/Swagger documentation
- Beautiful CLI with Rich formatting and interactive mode
- Batch processing support (up to 500 texts)
- Result caching with TTLCache
- Rate limiting (60 req/min single, 20 req/min batch)
- Comprehensive test suite
- Docker and docker-compose support
- Health check endpoints
- Middleware for request logging and timing

### Changed
- Refactored engine into separate module
- Added detailed docstrings and type hints
- Improved error handling
- Enhanced CLI with new commands

### Fixed
- Proper text validation for empty/whitespace input
- Cache key generation consistency

### Dependencies
- fastapi >= 0.104.0
- transformers >= 4.35.0
- torch >= 2.0.0
- click >= 8.1.0
- rich >= 13.0.0
- loguru >= 0.7.0

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Basic sentiment analysis
- Simple API endpoint
