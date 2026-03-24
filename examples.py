"""
Example usage of SentimentPulse
Built by Himal Badu, AI Founder
"""

# Example 1: Basic single text analysis
from sentimentpulse import analyze_sentiment

result = analyze_sentiment("I love this product! It's amazing!")
print(result)
# Output: {'sentiment': 'positive', 'score': 0.98, 'confidence': 0.99, ...}


# Example 2: Batch analysis
from sentimentpulse import analyze_batch

texts = [
    "This is absolutely wonderful!",
    "Terrible experience, would not recommend",
    "It's okay, nothing special",
]

results = analyze_batch(texts)
for text, result in zip(texts, results):
    print(f"{text}: {result['sentiment']}")


# Example 3: Using the engine directly
from sentimentpulse import get_engine

engine = get_engine()
engine.load_model()

# Single analysis
result = engine.analyze("Great job team!", verbose=True)
print(result)


# Example 4: Batch with progress
texts = [f"Review {i}: This is sample text" for i in range(100)]
results = engine.analyze_batch(
    texts, 
    use_cache=True,
    verbose=False,
    show_progress=True
)
print(f"Processed {len(results)} texts")


# Example 5: Cache statistics
stats = engine.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']}%")


# Example 6: Health check
health = engine.health_check()
print(f"Status: {health['status']}")
print(f"Model: {health['model_name']}")
print(f"Device: {health['device']}")


# Example 7: Using configuration
from sentimentpulse.config import get_settings

settings = get_settings()
print(f"API Version: {settings.api_version}")
print(f"Port: {settings.port}")


# Example 8: Lightweight analyzer (no engine loading)
from sentimentpulse.utils import SentimentAnalyzer

analyzer = SentimentAnalyzer(use_cache=True)
result = analyzer.analyze("Quick analysis")
print(result)


# Example 9: Validation utilities
from sentimentpulse.utils import validate_text_input, calculate_sentiment_distribution

# Validate input
is_valid, error = validate_text_input("Hello world")
print(f"Valid: {is_valid}")

# Calculate distribution
results = [
    {"sentiment": "positive"},
    {"sentiment": "positive"},
    {"sentiment": "negative"},
]
distribution = calculate_sentiment_distribution(results)
print(distribution)
