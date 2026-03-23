"""
SentimentPulse Package Setup
Built by Himal Badu, AI Founder
"""

from setuptools import setup, find_packages

setup(
    name="sentimentpulse",
    version="2.0.0",
    description="Production-grade sentiment analysis API and CLI powered by transformers",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    author="Himal Badu",
    author_email="himal@example.com",
    url="https://github.com/Himal-Badu/SentimentPulse",
    packages=find_packages(),
    install_requires=[
        # Core
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        
        # ML/AI - Transformers
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "accelerate>=0.24.0",
        
        # NLP utilities
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        
        # Rate limiting & Caching
        "slowapi>=0.1.8",
        "redis>=5.0.0",
        "cachetools>=5.3.0",
        
        # CLI
        "click>=8.1.0",
        "rich>=13.0.0",
        
        # Logging & Monitoring
        "loguru>=0.7.0",
        "sentry-sdk[fastapi]>=1.38.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.25.0",
            "pytest-cov>=4.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "sentiment=cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Artificial Intelligence :: Natural Language Processing :: Sentiment Analysis",
    ],
    python_requires=">=3.10",
)
