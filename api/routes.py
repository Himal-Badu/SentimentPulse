"""
API Routes for SentimentPulse
Built by Himal Badu, AI Founder
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from sentimentpulse import analyzer

router = APIRouter(prefix="/api/v1", tags=["analysis"])


class AnalyzeRequest(BaseModel):
    """Request model for single text analysis."""
    text: str = Field(..., min_length=1, max_length=10000)
    verbose: bool = False


class BatchRequest(BaseModel):
    """Request model for batch analysis."""
    texts: List[str] = Field(..., min_length=1, max_length=1000)
    verbose: bool = False


class AnalyzeResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str
    score: float
    confidence: float
    raw_scores: Optional[dict] = None


class BatchResponse(BaseModel):
    """Response model for batch analysis."""
    results: List[AnalyzeResponse]
    total: int
    processed_at: str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze sentiment of a single text.
    
    Returns sentiment (positive/negative/neutral), score, and confidence.
    """
    try:
        result = analyzer.analyze(request.text, request.verbose)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch", response_model=BatchResponse)
async def analyze_batch(request: BatchRequest):
    """
    Analyze sentiment of multiple texts.
    
    Returns array of results with total count.
    """
    try:
        results = analyzer.analyze_batch(request.texts, request.verbose)
        return {
            "results": results,
            "total": len(results),
            "processed_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze")
async def analyze_get():
    """GET endpoint - use POST for analysis."""
    return {"message": "Use POST /api/v1/analyze with JSON body"}
