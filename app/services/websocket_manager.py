from fastapi import WebSocket
import logging
from datetime import datetime
from typing import Dict, Set

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}
        self.pending_questions = {}
        self.answers = {}

    def add_connection(self, websocket: WebSocket):
        """Add a WebSocket connection"""
        self.active_connections[id(websocket)] = websocket
        logger.debug(f"Added WebSocket connection. Active connections: {len(self.active_connections)}")

    def remove_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if id(websocket) in self.active_connections:
            del self.active_connections[id(websocket)]
            logger.debug(f"Removed WebSocket connection. Active connections: {len(self.active_connections)}")

    @property
    def has_active_connections(self) -> bool:
        """Check if there are any active connections"""
        return len(self.active_connections) > 0

    async def send_progress(self, session_id: str, progress: int, current_step: str):
        """Send progress update to all connected clients"""
        message = {
            "type": "PROGRESS",
            "payload": {
                "progress": progress,
                "currentStep": current_step
            }
        }
        await self._broadcast(message)

    async def send_question(self, session_id: str, question_id: str, content: str, options: list):
        """Send a question to all connected clients"""
        message = {
            "type": "QUESTION",
            "payload": {
                "questionId": question_id,
                "content": content,
                "options": options
            }
        }
        self.pending_questions[question_id] = True
        await self._broadcast(message)

    async def send_error(self, session_id: str, error_message: str):
        """Send an error message to all connected clients"""
        message = {
            "type": "ERROR",
            "payload": {
                "message": error_message
            }
        }
        await self._broadcast(message)

    async def _broadcast(self, message):
        """Send a message to all connected clients"""
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")

    async def wait_for_answer(self, question_id: str, timeout: float = 30.0) -> str:
        """Wait for an answer to a specific question"""
        import asyncio
        
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if question_id in self.answers:
                answer = self.answers[question_id]
                del self.answers[question_id]
                if question_id in self.pending_questions:
                    del self.pending_questions[question_id]
                return answer
            await asyncio.sleep(0.1)
            
        # If we timeout, clean up
        if question_id in self.pending_questions:
            del self.pending_questions[question_id]
        return None

    async def handle_answer(self, answer):
        """Handle an answer from a client"""
        logger.info(f"Received answer for question {answer.question_id}: {answer.selected_option}")
        self.answers[answer.question_id] = answer.selected_option
        if answer.question_id in self.pending_questions:
            del self.pending_questions[answer.question_id]
