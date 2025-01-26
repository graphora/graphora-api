from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict, Any
from uuid import uuid4
import logging
from pydantic import BaseModel

from app.services.websocket_manager import WebSocketManager
from app.schemas.merge_events import MergeAnswer
from app.services.merge_service import MergeService
from app.dependencies import get_merge_service

router = APIRouter(prefix="/api/v1/merge", tags=["Merge"])
ws_manager = WebSocketManager()

logger = logging.getLogger(__name__)

class StartMergeRequest(BaseModel):
    transform_id: str

@router.websocket("/ws/{session_id}")
async def merge_websocket(
    websocket: WebSocket,
    session_id: str,
    transform_id: str | None = None,
    merge_service: MergeService = Depends(get_merge_service)
):
    """WebSocket endpoint for merge process"""
    try:
        # Connect the WebSocket
        await ws_manager.connect(websocket, session_id)
        
        # Associate the WebSocket manager with the merge service
        merge_service.set_websocket_manager(ws_manager, session_id)
        
        try:
            while True:
                # Wait for messages from the client
                message = await websocket.receive_json()
                
                # Handle different message types
                if message.get("type") == "ANSWER":
                    answer = MergeAnswer(**message.get("payload", {}))
                    await merge_service.handle_answer(answer)
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {str(e)}")
            await ws_manager.send_error(session_id, str(e))
            
    finally:
        ws_manager.disconnect(websocket, session_id)
        
@router.post("/{session_id}/start", response_model=Dict[str, str])
async def start_merge(
    session_id: str,
    request: StartMergeRequest,
    merge_service: MergeService = Depends(get_merge_service)
):
    """Start a new merge process"""
    await merge_service.start_merge(session_id, f"Staging_{request.transform_id}")
    return {"sessionId": session_id}

@router.post("/{session_id}/cancel", response_model=Dict[str, str])
async def cancel_merge(
    session_id: str,
    merge_service: MergeService = Depends(get_merge_service)
):
    """Cancel an ongoing merge process"""
    await merge_service.cancel_merge(session_id)
    return {"status": "cancelled"}
