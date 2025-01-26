from typing import Dict, Set
from fastapi import WebSocket
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        logger.info(f"New WebSocket connection for session {session_id}")
        
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Disconnect a WebSocket client"""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")
        
    async def broadcast_to_session(self, message: dict, session_id: str):
        """Broadcast a message to all connections in a session"""
        if session_id not in self.active_connections:
            return
            
        # Add timestamp if not present
        if 'timestamp' not in message:
            message['timestamp'] = datetime.utcnow().isoformat()
            
        dead_connections = set()
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                dead_connections.add(connection)
                
        # Clean up dead connections
        for dead in dead_connections:
            self.active_connections[session_id].discard(dead)
            
    async def send_progress(self, session_id: str, progress: float, current_step: str, graph_data=None):
        """Send a progress update event"""
        await self.broadcast_to_session({
            "type": "PROGRESS",
            "payload": {
                "data": {
                    "progress": progress,
                    "currentStep": current_step,
                    "graphData": graph_data
                }
            }
        }, session_id)
        
    async def send_question(self, session_id: str, question_id: str, content: str, 
                          options: list, preview_graph_data=None):
        """Send a question event"""
        await self.broadcast_to_session({
            "type": "QUESTION",
            "payload": {
                "data": {
                    "questionId": question_id,
                    "content": content,
                    "options": options,
                    "previewGraphData": preview_graph_data
                }
            }
        }, session_id)
        
    async def send_error(self, session_id: str, error_message: str):
        """Send an error event"""
        await self.broadcast_to_session({
            "type": "ERROR",
            "payload": {
                "data": {
                    "message": error_message
                }
            }
        }, session_id)
