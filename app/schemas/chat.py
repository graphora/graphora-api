"""Chat and Schema Refinement Pydantic Models"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Chat Session Models
class StartSessionRequest(BaseModel):
    """Request model for starting a new chat session"""
    context_type: str = Field(..., description="Type of chat context")
    context_id: Optional[str] = Field(None, description="ID of related resource")
    initial_context: Optional[Dict[str, Any]] = Field(None, description="Initial context data")
    title: Optional[str] = Field(None, description="Session title")

class StartSessionResponse(BaseModel):
    """Response model for session creation"""
    session_id: str = Field(..., description="Generated session ID")
    status: str = Field(..., description="Operation status")

class SendMessageRequest(BaseModel):
    """Request model for sending a message to a chat session"""
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., description="User message content")
    message_type: str = Field(default="user_message", description="Type of message")

class ChatMessageResponse(BaseModel):
    """Response model for individual chat messages"""
    id: str = Field(..., description="Message ID")
    type: str = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ChatSessionResponse(BaseModel):
    """Response model for chat session data"""
    session_id: str = Field(..., description="Session ID")
    session_context: Dict[str, Any] = Field(..., description="Session context")
    messages: List[ChatMessageResponse] = Field(..., description="Chat messages")
    total_messages: int = Field(..., description="Total message count")

class UserSessionsResponse(BaseModel):
    """Response model for user's chat sessions"""
    sessions: List[Dict[str, Any]] = Field(..., description="User's chat sessions")

class EndSessionRequest(BaseModel):
    """Request model for ending a chat session"""
    summary: Optional[str] = Field(None, description="Session summary")

# Schema Refinement Models
class SchemaRefinementRequest(BaseModel):
    """Request model for schema refinement"""
    session_id: Optional[str] = Field(None, description="Existing session ID")
    initial_schema: Optional[str] = Field(None, description="Initial schema (for new session)")
    schema_id: Optional[str] = Field(None, description="Related schema ID")
    user_request: str = Field(..., description="Refinement request")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "user_request": "Add timestamps to all entities",
                "session_id": "uuid-string",
                "context": {"domain": "healthcare"}
            }
        }

class SchemaRefinementResponse(BaseModel):
    """Response model for schema refinement"""
    success: bool = Field(..., description="Operation success")
    session_id: str = Field(..., description="Chat session ID")
    message_id: Optional[str] = Field(None, description="Response message ID")
    refined_schema: Optional[str] = Field(None, description="Refined schema content")
    changes_made: Optional[List[str]] = Field(None, description="List of changes made")
    confidence: Optional[float] = Field(None, description="Confidence score")
    response_content: str = Field(..., description="Assistant response")
    error: Optional[str] = Field(None, description="Error message if failed")

class GetCurrentSchemaResponse(BaseModel):
    """Response model for getting current schema state"""
    session_id: str = Field(..., description="Session ID")
    current_schema: str = Field(..., description="Current schema content")

# Common Response Models
class OperationResponse(BaseModel):
    """Generic operation response"""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")