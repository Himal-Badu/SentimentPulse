"""
SentimentPulse - WebSocket support for real-time analysis
Built by Himal Badu, AI Founder
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, WebSocketException
from pydantic import BaseModel, Field
import json
import asyncio


# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and track new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


# Create manager and router
ws_manager = ConnectionManager()
ws_router = APIRouter()


# Request/Response models for WebSocket
class WSAnalyzeRequest(BaseModel):
    """WebSocket request for analysis."""
    text: str = Field(..., min_length=1, max_length=10000)
    request_id: Optional[str] = Field(None, description="Request identifier")
    use_cache: bool = Field(True, description="Use caching")
    verbose: bool = Field(False, description="Include detailed scores")


class WSAnalyzeResponse(BaseModel):
    """WebSocket response for analysis."""
    request_id: Optional[str] = None
    sentiment: str
    score: float
    confidence: float
    model: str
    analyzed_at: str
    raw_scores: Optional[Dict[str, Any]] = None


class WSBatchRequest(BaseModel):
    """WebSocket request for batch analysis."""
    texts: List[str] = Field(..., min_length=1, max_length=100)
    request_id: Optional[str] = None
    use_cache: bool = True
    verbose: bool = False


class WSMessage(BaseModel):
    """Base WebSocket message."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# WebSocket endpoints
@ws_router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """WebSocket endpoint for real-time sentiment analysis."""
    await ws_manager.connect(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                request_data = json.loads(data)
            except json.JSONDecodeError:
                await ws_manager.send_personal_message({
                    "type": "error",
                    "error": "Invalid JSON",
                    "data": {}
                }, websocket)
                continue
            
            # Handle different message types
            msg_type = request_data.get("type", "analyze")
            
            if msg_type == "analyze":
                # Single text analysis
                text = request_data.get("text", "")
                use_cache = request_data.get("use_cache", True)
                verbose = request_data.get("verbose", False)
                request_id = request_data.get("request_id")
                
                if not text:
                    await ws_manager.send_personal_message({
                        "type": "error",
                        "error": "Text is required",
                        "request_id": request_id
                    }, websocket)
                    continue
                
                # Analyze
                from sentimentpulse import analyze_sentiment
                result = analyze_sentiment(text, use_cache=use_cache, verbose=verbose)
                
                # Send response
                await ws_manager.send_personal_message({
                    "type": "result",
                    "request_id": request_id,
                    "data": result
                }, websocket)
            
            elif msg_type == "batch":
                # Batch analysis
                texts = request_data.get("texts", [])
                request_id = request_data.get("request_id")
                
                if not texts:
                    await ws_manager.send_personal_message({
                        "type": "error",
                        "error": "Texts array is required",
                        "request_id": request_id
                    }, websocket)
                    continue
                
                # Send processing status
                await ws_manager.send_personal_message({
                    "type": "status",
                    "message": f"Processing {len(texts)} texts...",
                    "request_id": request_id
                }, websocket)
                
                # Analyze batch
                from sentimentpulse import analyze_batch
                results = analyze_batch(texts, use_cache=True, verbose=False)
                
                # Send results
                await ws_manager.send_personal_message({
                    "type": "batch_result",
                    "request_id": request_id,
                    "data": {
                        "results": results,
                        "total": len(results)
                    }
                }, websocket)
            
            elif msg_type == "ping":
                # Keepalive ping
                await ws_manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket)
            
            else:
                await ws_manager.send_personal_message({
                    "type": "error",
                    "error": f"Unknown message type: {msg_type}",
                    "request_id": request_data.get("request_id")
                }, websocket)
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        await ws_manager.send_personal_message({
            "type": "error",
            "error": str(e)
        }, websocket)
        ws_manager.disconnect(websocket)


@ws_router.websocket("/ws/health")
async def websocket_health(websocket: WebSocket):
    """WebSocket endpoint for health status updates."""
    await ws_manager.connect(websocket)
    
    try:
        while True:
            # Get health status
            from sentimentpulse.monitoring import get_health_monitor
            monitor = get_health_monitor()
            health = monitor.get_health_status()
            
            await ws_manager.send_personal_message({
                "type": "health",
                "data": health
            }, websocket)
            
            # Wait before next update
            await asyncio.sleep(5)
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
