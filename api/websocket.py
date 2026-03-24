"""
SentimentPulse - WebSocket API for real-time streaming
Built by Himal Badu, AI Founder

WebSocket endpoint for real-time sentiment analysis with streaming support.
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from loguru import logger

from sentimentpulse import get_engine
from sentimentpulse.utils import validate_text_input


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and track new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def send_message(self, message: Dict[str, Any], client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        for client_id in list(self.active_connections.keys()):
            try:
                await self.send_message(message, client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                self.disconnect(client_id)


# Global connection manager
manager = ConnectionManager()

# WebSocket router
router = APIRouter()


@router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sentiment analysis.
    
    Clients send text messages and receive analysis results in real-time.
    """
    import uuid
    
    client_id = str(uuid.uuid4())[:8]
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        await manager.send_message({
            "type": "connected",
            "client_id": client_id,
            "message": "Connected to SentimentPulse WebSocket"
        }, client_id)
        
        # Ensure model is loaded
        engine = get_engine()
        if not engine._model_loaded:
            await manager.send_message({
                "type": "status",
                "message": "Loading model..."
            }, client_id)
            engine.load_model()
            await manager.send_message({
                "type": "status",
                "message": "Model loaded"
            }, client_id)
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_text()
            
            try:
                # Parse incoming data
                message = json.loads(data)
                
                # Support both simple text and structured messages
                if isinstance(message, dict):
                    text = message.get("text", "")
                    request_id = message.get("request_id", str(uuid.uuid4())[:8])
                else:
                    text = str(message)
                    request_id = str(uuid.uuid4())[:8]
                
                # Validate input
                is_valid, error = validate_text_input(text)
                if not is_valid:
                    await manager.send_message({
                        "type": "error",
                        "request_id": request_id,
                        "error": error
                    }, client_id)
                    continue
                
                # Analyze sentiment
                result = engine.analyze(text)
                
                # Send result
                await manager.send_message({
                    "type": "result",
                    "request_id": request_id,
                    "sentiment": result["sentiment"],
                    "score": result["score"],
                    "confidence": result["confidence"],
                    "model": result["model"],
                    "analyzed_at": result["analyzed_at"]
                }, client_id)
                
            except json.JSONDecodeError:
                # Treat plain text as sentiment request
                is_valid, error = validate_text_input(data)
                if not is_valid:
                    await manager.send_message({
                        "type": "error",
                        "error": error
                    }, client_id)
                    continue
                
                result = engine.analyze(data)
                await manager.send_message({
                    "type": "result",
                    "sentiment": result["sentiment"],
                    "score": result["score"],
                    "confidence": result["confidence"]
                }, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await manager.send_message({
            "type": "error",
            "error": str(e)
        }, client_id)
        manager.disconnect(client_id)


@router.websocket("/ws/batch")
async def websocket_batch(websocket: WebSocket):
    """
    WebSocket endpoint for batch sentiment analysis with streaming results.
    """
    import uuid
    
    client_id = str(uuid.uuid4())[:8]
    await manager.connect(websocket, client_id)
    
    try:
        await manager.send_message({
            "type": "connected",
            "client_id": client_id,
            "message": "Connected to SentimentPulse Batch WebSocket"
        }, client_id)
        
        engine = get_engine()
        if not engine._model_loaded:
            engine.load_model()
        
        # Receive batch request
        data = await websocket.receive_text()
        request_data = json.loads(data)
        
        texts = request_data.get("texts", [])
        
        if not texts:
            await manager.send_message({
                "type": "error",
                "error": "No texts provided"
            }, client_id)
            return
        
        # Process and stream results
        total = len(texts)
        
        for i, text in enumerate(texts):
            is_valid, error = validate_text_input(text)
            
            if not is_valid:
                await manager.send_message({
                    "type": "result",
                    "index": i,
                    "error": error
                }, client_id)
                continue
            
            result = engine.analyze(text)
            await manager.send_message({
                "type": "result",
                "index": i,
                "sentiment": result["sentiment"],
                "score": result["score"],
                "confidence": result["confidence"],
                "progress": f"{i+1}/{total}"
            }, client_id)
        
        # Send completion message
        await manager.send_message({
            "type": "complete",
            "total": total
        }, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket batch error for {client_id}: {e}")
        await manager.send_message({
            "type": "error",
            "error": str(e)
        }, client_id)
        manager.disconnect(client_id)


async def notify_all_clients(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients."""
    await manager.broadcast(message)
