"""
SentimentPulse Package Setup
Built by Himal Badu, AI Founder
"""

from setuptools import setup, find_packages

setup(
    name="sentimentpulse",
    version="1.0.0",
    description="Real-time sentiment analysis API and CLI tool",
    author="Himal Badu",
    author_email="himal@example.com",
    url="https://github.com/Himal-Badu/SentimentPulse",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "click>=8.1.0",
        "nltk>=3.8.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "httpx>=0.25.0",
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
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
)
