from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
from typing import Dict, Any
import logging
from pydantic import BaseModel
import asyncio
from app.schemas.merge_events import MergeAnswer
from app.services.merge_service import MergeService
from app.dependencies import get_merge_service
from app.schemas.global_merge import ResolutionStatus
from app.config import settings
from app.utils.mock import transform_id as mock_transform_id

router = APIRouter(prefix="/api/v1/merge", tags=["Merge"])
logger = logging.getLogger(__name__)

class StartMergeRequest(BaseModel):
    transform_id: str

@router.websocket("/ws/{session_id}")
async def merge_websocket(
    websocket: WebSocket,
    session_id: str,
    merge_service: MergeService = Depends(get_merge_service)
):
    """WebSocket endpoint for merge process"""
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for session {session_id}")
        await merge_service.handle_websocket_connection(session_id, websocket)
        
        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"Received WebSocket message: {data}")
                if data.get("type") == "ANSWER":
                    answer = MergeAnswer(**data.get("payload", {}))
                    await merge_service.handle_answer(answer)
            except StarletteWebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {str(e)}")
                if not websocket.client_state.DISCONNECTED:
                    await websocket.send_json({
                        "type": "ERROR",
                        "payload": {"message": str(e)}
                    })
                break
                
    except StarletteWebSocketDisconnect:
        logger.info(f"WebSocket disconnected during setup for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
    finally:
        logger.info(f"Cleaning up WebSocket for session {session_id}")
        if session_id in merge_service.ws_managers:
            merge_service.ws_managers[session_id].remove_connection(websocket)
            if not merge_service.ws_managers[session_id].active_connections:
                await merge_service.cancel_merge(session_id)

@router.post("/{session_id}/start", response_model=Dict[str, str])
async def start_merge(
    session_id: str,
    request: StartMergeRequest,
    merge_service: MergeService = Depends(get_merge_service)
):
    """Start a new merge process"""
    # Wait for WebSocket connection with retries
    max_retries = 3
    retry_delay = 1.0  # seconds
    
    for attempt in range(max_retries):
        if merge_service.is_websocket_ready(session_id):
            break
            
        if attempt < max_retries - 1:
            logger.info(f"Waiting for WebSocket connection, attempt {attempt + 1}/{max_retries}")
            await asyncio.sleep(retry_delay)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"WebSocket connection not established after {max_retries} attempts. Please ensure WebSocket is connected before starting merge."
            )
    
    logger.info(f"Starting merge for session {session_id}")
    if settings.MOCK_MODE:
        logger.info("Mock mode enabled, skipping document processing")
        request.transform_id = mock_transform_id
    await merge_service.start_merge(session_id, f"Staging_{request.transform_id}")
    return {"sessionId": session_id, "status": "started"}

@router.post("/{session_id}/cancel", response_model=Dict[str, str])
async def cancel_merge(
    session_id: str,
    merge_service: MergeService = Depends(get_merge_service)
):
    """Cancel an ongoing merge process"""
    await merge_service.cancel_merge(session_id)
    return {"status": "cancelled"}

@router.get("/{session_id}/visualization")
async def get_merge_visualization(session_id: str, merge_service: MergeService = Depends(get_merge_service)):
    """Get visualization data for merge changes and conflicts"""
    try:
        # Get the current state for this session
        state = await merge_service.get_session_state(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Merge session not found")
            
        # Get visualization data
        viz_data = state.get_visualization_data()
        
        return {
            "status": "success",
            "data": {
                "nodes": viz_data["nodes"],
                "edges": viz_data["edges"],
                "conflicts": viz_data["conflicts"],
                "summary": {
                    "total_nodes": len(viz_data["nodes"]),
                    "new_nodes": len(state.new_nodes),
                    "updated_nodes": len(state.updated_nodes),
                    "conflicts": len(state.conflicts),
                    "status": {
                        "new": len([n for n in state.processed_nodes if n.status == ResolutionStatus.NEW]),
                        "resolved": len([n for n in state.processed_nodes if n.status == ResolutionStatus.RESOLVED]),
                        "needs_review": len([n for n in state.processed_nodes if n.status == ResolutionStatus.NEEDS_REVIEW])
                    }
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting merge visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
