"""
SentimentPulse - Core Sentiment Analyzer
Built by Himal Badu, AI Founder
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, List, Optional
import os

# Download VADER lexicon if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    """Core sentiment analysis using VADER."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str, verbose: bool = False) -> Dict:
        """
        Analyze text sentiment.
        
        Args:
            text: Input text to analyze
            verbose: Include detailed scores
            
        Returns:
            Dictionary with sentiment results
        """
        if not text or not text.strip():
            return self._empty_result()
        
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Determine sentiment label
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Normalize confidence (compound score magnitude)
        confidence = abs(compound)
        
        result = {
            "sentiment": sentiment,
            "score": round(compound, 4),
            "confidence": round(confidence, 4),
            "raw_scores": {
                "pos": round(scores['pos'], 4),
                "neg": round(scores['neg'], 4),
                "neu": round(scores['neu'], 4)
            }
        }
        
        if not verbose:
            result.pop('raw_scores', None)
        
        return result
    
    def analyze_batch(self, texts: List[str], verbose: bool = False) -> List[Dict]:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of texts to analyze
            verbose: Include detailed scores
            
        Returns:
            List of sentiment results
        """
        return [self.analyze(text, verbose) for text in texts]
    
    def _empty_result(self) -> Dict:
        """Return empty result for invalid input."""
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "raw_scores": {"pos": 0.0, "neg": 0.0, "neu": 1.0}
        }


# Singleton instance
analyzer = SentimentAnalyzer()


def analyze_sentiment(text: str, verbose: bool = False) -> Dict:
    """Convenience function for sentiment analysis."""
    return analyzer.analyze(text, verbose)


def analyze_batch(texts: List[str], verbose: bool = False) -> List[Dict]:
    """Convenience function for batch analysis."""
    return analyzer.analyze_batch(texts, verbose)
