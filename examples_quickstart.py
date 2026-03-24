"""
SentimentPulse - Quick Start Examples
Built by Himal Badu, AI Founder

This module contains simple examples to help you get started with SentimentPulse.
"""

from sentimentpulse import analyze_sentiment, analyze_batch


def example_single_analysis():
    """Example: Analyze a single text."""
    result = analyze_sentiment("I absolutely love this product! It's amazing!")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Score: {result['score']}")
    print(f"Confidence: {result['confidence']}")
    return result


def example_batch_analysis():
    """Example: Analyze multiple texts."""
    texts = [
        "This product is great!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
        "Absolutely love it!",
        "Worst purchase ever."
    ]
    
    results = analyze_batch(texts)
    
    for text, result in zip(texts, results):
        print(f"'{text}' -> {result['sentiment']} ({result['score']:+.2f})")
    
    return results


def example_with_verbose():
    """Example: Get detailed model scores."""
    result = analyze_sentiment(
        "This is an interesting product with some pros and cons.",
        verbose=True
    )
    print(f"Raw scores: {result.get('raw_scores')}")
    return result


def example_batch_with_stats():
    """Example: Analyze batch with distribution statistics."""
    from sentimentpulse.utils import calculate_sentiment_distribution
    
    texts = [
        "Love it!",
        "Hate it!",
        "Neutral",
        "Great!",
        "Awful!",
        "Fine"
    ]
    
    results = analyze_batch(texts)
    stats = calculate_sentiment_distribution(results)
    
    print(f"Total: {stats['total']}")
    print(f"Positive: {stats['positive']} ({stats['positive_pct']}%)")
    print(f"Negative: {stats['negative']} ({stats['negative_pct']}%)")
    print(f"Neutral: {stats['neutral']} ({stats['neutral_pct']}%)")
    
    return stats


def example_caching():
    """Example: Using cache for better performance."""
    import time
    
    text = "This is a cached analysis example"
    
    # First call - cache miss
    start = time.time()
    result1 = analyze_sentiment(text, use_cache=True)
    time1 = time.time() - start
    
    # Second call - cache hit
    start = time.time()
    result2 = analyze_sentiment(text, use_cache=True)
    time2 = time.time() - start
    
    print(f"First call: {time1*1000:.2f}ms")
    print(f"Second call (cached): {time2*1000:.2f}ms")
    print(f"Speedup: {time1/time2:.1f}x")
    
    return result1, result2


def example_custom_engine():
    """Example: Using custom engine instance."""
    from sentimentpulse import SentimentEngine
    
    # Create custom engine
    engine = SentimentEngine()
    engine.load_model()
    
    # Analyze with custom engine
    result = engine.analyze("Custom engine analysis example")
    print(f"Result: {result}")
    
    return result


if __name__ == "__main__":
    print("=== SentimentPulse Examples ===\n")
    
    print("1. Single Analysis:")
    example_single_analysis()
    print()
    
    print("2. Batch Analysis:")
    example_batch_analysis()
    print()
    
    print("3. With Verbose:")
    example_with_verbose()
    print()
    
    print("4. Batch with Stats:")
    example_batch_with_stats()
    print()
    
    print("5. Caching:")
    example_caching()
    print()
    
    print("6. Custom Engine:")
    example_custom_engine()
