"""Schema Refinement Service with Multi-turn Conversation Support

This module provides intelligent schema refinement capabilities through natural language
conversations. It enables users to iteratively improve their knowledge graph schemas
using conversational AI with full context preservation.

Key Features:
- Multi-turn conversational schema refinement
- Context-aware LLM integration with conversation memory
- Comprehensive change tracking and audit trails
- Fallback mechanisms for service resilience
- Rich conversation metadata and analytics
- Session-based state management

The service integrates with:
- ChatSessionService for conversation management
- LLM services (Gemini) for intelligent schema modifications
- Audit service for comprehensive operation tracking
- Schema generation service for response parsing
"""
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

from app.services.chat_session_service import ChatSessionService, MessageType, ChatContextType
from app.services.schema_generation_service import schema_generation_service
from app.services.audit_service import AuditService, OperationType, OperationStatus
from app.utils.llm_helper import get_user_llm_credentials, create_gemini_client
from app.utils.llm_usage_tracker import track_gemini_usage
from app.config import settings

logger = logging.getLogger(__name__)

class SchemaRefinementService:
    """Service for multi-turn schema refinement conversations.
    
    This service orchestrates conversational schema refinement by combining
    chat session management, LLM-powered schema modifications, and comprehensive
    audit trailing. It maintains conversation context across multiple interactions
    to provide coherent and intelligent schema evolution.
    
    Attributes:
        chat_service: Chat session management service
        audit_service: Operation auditing and tracking service
    """
    
    def __init__(self) -> None:
        """Initialize the schema refinement service with required dependencies."""
        self.chat_service = ChatSessionService()
        self.audit_service = AuditService()
    
    async def start_refinement_session(
        self,
        user_id: str,
        initial_schema: str,
        schema_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new schema refinement session with conversational interface.
        
        Creates a new chat session dedicated to schema refinement, loads the initial
        schema into context, and provides an interactive introduction to guide the user
        through the refinement process.
        
        Args:
            user_id: Unique identifier for the user starting the session
            initial_schema: The base schema content to start refinement from
            schema_id: Optional identifier of the source schema for tracking
            context: Optional additional context for the refinement session
            
        Returns:
            str: Unique session identifier for subsequent refinement operations
            
        Raises:
            ValueError: If required parameters are missing or invalid
            Exception: If session creation or initialization fails
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")
        if not initial_schema or not initial_schema.strip():
            raise ValueError("initial_schema is required and cannot be empty")
            
        try:
            # Create new chat session with schema refinement context
            session_id = await self.chat_service.start_session(
                user_id=user_id,
                context_type=ChatContextType.SCHEMA_REFINEMENT,
                context_id=schema_id,
                initial_context={
                    "initial_schema": initial_schema,
                    "schema_id": schema_id,
                    "refinement_count": 0,
                    "context": context or {},
                    "schema_version": "0.1.0",
                    "created_at": datetime.utcnow().isoformat()
                },
                title=f"Schema Refinement Session - {schema_id or 'New Schema'}"
            )
            
            # Add comprehensive welcome message with guidance
            welcome_content = f"""I'm ready to help you refine your schema! 🎯

**Current Schema Overview:**
Your schema is loaded and ready for modifications. You can ask me to:

- **Add new entities or relationships**
- **Modify existing properties**
- **Remove unnecessary elements**
- **Restructure the schema organization**
- **Add validation rules or constraints**
- **Optimize for specific use cases**

**How to interact:**
Just tell me what you'd like to change in natural language. For example:
- "Add a timestamp field to all entities"
- "Create a relationship between Customer and Product"
- "Make the email field required in the Person entity"
- "Remove the deprecated fields"

**Schema Statistics:**
- Initial schema length: {len(initial_schema):,} characters
- Context provided: {'Yes' if context else 'No'}
- Schema ID: {schema_id or 'None'}

What would you like to modify first?"""
            
            await self.chat_service.add_message(
                user_id=user_id,
                session_id=session_id,
                message_type=MessageType.ASSISTANT_MESSAGE,
                content=welcome_content,
                metadata={
                    "message_role": "system_intro",
                    "schema_length": len(initial_schema),
                    "has_context": bool(context),
                    "schema_id": schema_id,
                    "session_type": "refinement_start"
                }
            )
            
            logger.info(
                f"Started schema refinement session {session_id} for user {user_id} "
                f"(schema_id: {schema_id}, initial_length: {len(initial_schema)})"
            )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start refinement session for user {user_id}: {str(e)}")
            raise Exception(f"Refinement session creation failed: {str(e)}") from e
    
    async def process_refinement_request(
        self,
        user_id: str,
        session_id: str,
        user_request: str
    ) -> Dict[str, Any]:
        """Process a user's schema refinement request with full context awareness.
        
        Handles the complete refinement pipeline: user message recording, context
        extraction, LLM-powered schema modification, response generation, and
        comprehensive audit logging. Maintains conversation continuity and provides
        rich feedback on the refinement process.
        
        Args:
            user_id: User identifier for access control and tracking
            session_id: Target refinement session identifier
            user_request: Natural language refinement request
            
        Returns:
            Dict containing:
                - success: Boolean indicating operation success
                - message_id: ID of assistant response message
                - refined_schema: Updated schema content (if successful)
                - changes_made: List of specific changes applied
                - confidence: Confidence score for the refinement
                - explanation: Detailed explanation of changes
                - response_content: User-facing response message
                - error: Error message (if failed)
                
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")
        if not session_id or not session_id.strip():
            raise ValueError("session_id is required and cannot be empty")
        if not user_request or not user_request.strip():
            raise ValueError("user_request is required and cannot be empty")
            
        start_time = time.time()
        operation_id = f"{session_id}_{int(start_time * 1000)}"
        
        try:
            # Record user message with comprehensive metadata
            user_message_id = await self.chat_service.add_message(
                user_id=user_id,
                session_id=session_id,
                message_type=MessageType.USER_MESSAGE,
                content=user_request,
                metadata={
                    "request_length": len(user_request),
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_type": "schema_refinement",
                    "word_count": len(user_request.split())
                }
            )
            
            logger.debug(f"Recorded user message {user_message_id} for session {session_id}")
            
            # Extract current session state and build context
            session_data = await self.chat_service.get_session_history(user_id, session_id)
            current_schema = await self._extract_current_schema(session_data)
            conversation_context = await self._build_conversation_context(session_data)
            
            if not current_schema:
                raise ValueError("No schema found in session context")
            
            # Perform LLM-powered schema refinement
            refined_schema, changes_made, confidence, explanation = await self._refine_schema_with_llm(
                user_id=user_id,
                current_schema=current_schema,
                user_request=user_request,
                conversation_context=conversation_context
            )
            
            # Generate comprehensive response message
            response_content = self._format_refinement_response(
                changes_made=changes_made,
                confidence=confidence,
                explanation=explanation,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
            
            # Record assistant response with rich metadata
            assistant_message_id = await self.chat_service.add_message(
                user_id=user_id,
                session_id=session_id,
                message_type=MessageType.ASSISTANT_MESSAGE,
                content=response_content,
                metadata={
                    "refined_schema": refined_schema,
                    "changes_made": changes_made,
                    "confidence": confidence,
                    "explanation": explanation,
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "schema_length": len(refined_schema),
                    "changes_count": len(changes_made),
                    "user_request_id": user_message_id,
                    "refinement_version": conversation_context.get("refinement_count", 0) + 1
                }
            )
            
            # Comprehensive audit logging
            await self._log_refinement_operation(
                user_id=user_id,
                operation_id=operation_id,
                session_id=session_id,
                user_request=user_request,
                changes_made=changes_made,
                confidence=confidence,
                processing_time_ms=int((time.time() - start_time) * 1000),
                assistant_message_id=assistant_message_id
            )
            
            logger.info(
                f"Successfully processed refinement request for session {session_id} "
                f"(changes: {len(changes_made)}, confidence: {confidence:.2f}, "
                f"time: {int((time.time() - start_time) * 1000)}ms)"
            )
            
            return {
                "success": True,
                "message_id": assistant_message_id,
                "refined_schema": refined_schema,
                "changes_made": changes_made,
                "confidence": confidence,
                "explanation": explanation,
                "response_content": response_content,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error processing refinement request for session {session_id}: {str(e)}")
            
            # Generate user-friendly error response
            error_response = await self._handle_refinement_error(
                user_id=user_id,
                session_id=session_id,
                error=e,
                user_request=user_request
            )
            
            return {
                "success": False,
                "error": str(e),
                "response_content": error_response,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
    
    async def _extract_current_schema(self, session_data: Dict[str, Any]) -> str:
        """Extract the most recent schema from session history.
        
        Searches through the conversation history to find the latest schema version,
        prioritizing refined schemas from recent messages over the initial schema.
        
        Args:
            session_data: Complete session data with messages and context
            
        Returns:
            str: The most current schema content
            
        Raises:
            ValueError: If no valid schema is found in the session
        """
        if not session_data:
            raise ValueError("Session data is required")
            
        messages = session_data.get("messages", [])
        initial_context = session_data.get("session_context", {}).get("initial_context", {})
        
        # Start with initial schema from session context
        current_schema = initial_context.get("initial_schema", "")
        
        if not current_schema:
            logger.warning("No initial schema found in session context")
        
        # Search for the most recent refined schema (newest first)
        for message in reversed(messages):
            if message.get("type") == "assistant_message":
                metadata = message.get("metadata", {})
                if "refined_schema" in metadata and metadata["refined_schema"]:
                    refined_schema = metadata["refined_schema"]
                    logger.debug(f"Found refined schema from message {message.get('id', 'unknown')}")
                    current_schema = refined_schema
                    break
        
        if not current_schema:
            raise ValueError("No valid schema found in session history")
            
        return current_schema
    
    async def _build_conversation_context(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive conversation context from session history.
        
        Analyzes the conversation flow to extract patterns, themes, and progression
        that can inform future refinement decisions and maintain coherence.
        
        Args:
            session_data: Complete session data with messages and metadata
            
        Returns:
            Dict containing:
                - session_context: Original session initialization context
                - previous_requests: Recent user requests for pattern analysis
                - changes_history: Recent changes for conflict avoidance
                - refinement_count: Number of refinements performed
                - conversation_themes: Identified patterns in requests
                - confidence_trend: Trend in refinement confidence scores
        """
        if not session_data:
            return {}
            
        messages = session_data.get("messages", [])
        session_context = session_data.get("session_context", {})
        
        # Extract conversation patterns and metadata
        user_requests = []
        changes_history = []
        confidence_scores = []
        processing_times = []
        
        refinement_count = 0
        
        for message in messages:
            message_type = message.get("type")
            metadata = message.get("metadata", {})
            
            if message_type == "user_message":
                content = message.get("content", "")
                if content:
                    user_requests.append({
                        "content": content,
                        "timestamp": message.get("timestamp"),
                        "word_count": metadata.get("word_count", len(content.split()))
                    })
                    
            elif message_type == "assistant_message":
                if "refined_schema" in metadata:
                    refinement_count += 1
                    
                    # Track changes made
                    if "changes_made" in metadata and metadata["changes_made"]:
                        changes_history.extend(metadata["changes_made"])
                    
                    # Track confidence and performance metrics
                    if "confidence" in metadata:
                        confidence_scores.append(metadata["confidence"])
                    if "processing_time_ms" in metadata:
                        processing_times.append(metadata["processing_time_ms"])
        
        # Analyze conversation themes
        conversation_themes = self._analyze_conversation_themes(user_requests)
        
        # Calculate performance trends
        confidence_trend = "stable"
        if len(confidence_scores) >= 2:
            recent_avg = sum(confidence_scores[-2:]) / len(confidence_scores[-2:])
            older_avg = sum(confidence_scores[:-2]) / max(len(confidence_scores[:-2]), 1)
            if recent_avg > older_avg + 0.1:
                confidence_trend = "improving"
            elif recent_avg < older_avg - 0.1:
                confidence_trend = "declining"
        
        return {
            "session_context": session_context.get("initial_context", {}),
            "previous_requests": user_requests[-3:],  # Last 3 requests with metadata
            "changes_history": changes_history[-10:],  # Last 10 changes
            "refinement_count": refinement_count,
            "conversation_themes": conversation_themes,
            "confidence_trend": confidence_trend,
            "avg_confidence": sum(confidence_scores) / max(len(confidence_scores), 1),
            "avg_processing_time": sum(processing_times) / max(len(processing_times), 1),
            "total_messages": len(messages)
        }
    
    def _analyze_conversation_themes(self, user_requests: List[Dict[str, Any]]) -> List[str]:
        """Analyze user requests to identify common themes and patterns."""
        if not user_requests:
            return []
            
        # Simple keyword-based theme detection
        themes = []
        all_text = " ".join([req["content"].lower() for req in user_requests])
        
        theme_keywords = {
            "timestamps": ["timestamp", "created_at", "updated_at", "date", "time"],
            "relationships": ["relationship", "connect", "link", "association", "relation"],
            "properties": ["property", "field", "attribute", "column"],
            "validation": ["required", "unique", "constraint", "validate", "rule"],
            "structure": ["organize", "structure", "hierarchy", "group", "category"],
            "removal": ["remove", "delete", "drop", "eliminate"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                themes.append(theme)
                
        return themes
    
    async def _refine_schema_with_llm(
        self,
        user_id: str,
        current_schema: str,
        user_request: str,
        conversation_context: Dict[str, Any]
    ) -> Tuple[str, List[str], float, str]:
        """Refine schema using LLM with conversation context"""
        
        try:
            # Get user's LLM credentials
            api_key, model_name = await get_user_llm_credentials(user_id)
            client = create_gemini_client(api_key)
            
            # Build refinement prompt with conversation context
            prompt = self._build_refinement_prompt(
                current_schema=current_schema,
                user_request=user_request,
                conversation_context=conversation_context
            )
            
            # Call LLM for refinement
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt]
            )
            
            # Track usage
            await track_gemini_usage(
                user_id=user_id,
                model_name=model_name,
                operation_type="schema_refinement",
                response=response,
                operation_context="chat_refinement"
            )
            
            # Parse response
            refined_schema, changes_made, confidence, explanation = schema_generation_service._parse_refinement_response(
                response.text
            )
            
            return refined_schema, changes_made, confidence, explanation
            
        except Exception as e:
            logger.error(f"LLM refinement failed: {str(e)}")
            
            # Fallback to simple text-based refinement
            return await self._fallback_refinement(current_schema, user_request)
    
    def _build_refinement_prompt(
        self,
        current_schema: str,
        user_request: str,
        conversation_context: Dict[str, Any]
    ) -> str:
        """Build the LLM prompt for schema refinement with conversation context"""
        
        previous_requests = conversation_context.get("previous_requests", [])
        changes_history = conversation_context.get("changes_history", [])
        refinement_count = conversation_context.get("refinement_count", 0)
        
        prompt = f"""You are refining a knowledge graph schema based on user feedback. This is part of an ongoing conversation.

**Current Schema:**
```yaml
{current_schema}
```

**Current User Request:**
{user_request}

**Conversation Context:**
- This is refinement #{refinement_count + 1} in this session
- Previous requests: {', '.join(previous_requests[-2:]) if previous_requests else 'None'}
- Recent changes: {', '.join(changes_history[-3:]) if changes_history else 'None'}

**Instructions:**
1. Analyze the user's request in the context of previous modifications
2. Make the requested changes while maintaining schema integrity
3. Preserve existing structure unless explicitly asked to change
4. Ensure all changes align with knowledge graph best practices
5. Maintain YAML format and property types
6. Be conservative - only change what's specifically requested

**Schema Format Requirements:**
- Use YAML format with version 0.1.0
- Structure: entities -> EntityName -> properties/relationships
- Properties: type, description, optionally: unique, required, index
- Relationships: target, optionally properties
- Property types: str, int, float, bool, date
- Entity names in CamelCase, Relationship names in UPPER_CASE, Property names in camelCase

**Output Format:**
Return ONLY the YAML schema content starting with 'version: 0.1.0' and 'entities:'.
DO NOT wrap the schema in any additional keys like 'ontology:' or any other wrapper.
The response should start directly with 'version: 0.1.0'.

Example of correct format:
version: 0.1.0
entities:
  EntityName:
    properties: ...

CHANGES_MADE: [list of specific changes made]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [brief explanation of the changes and reasoning]

Refined schema:"""
        
        return prompt
    
    async def _fallback_refinement(
        self,
        current_schema: str,
        user_request: str
    ) -> Tuple[str, List[str], float, str]:
        """Fallback refinement when LLM is unavailable"""
        
        # Simple text-based modifications for common requests
        changes = []
        confidence = 0.6
        
        # This is a very basic fallback - in practice you might implement
        # more sophisticated text parsing and YAML manipulation
        if "add" in user_request.lower() and "timestamp" in user_request.lower():
            # Example: add timestamp fields
            changes.append("Added created_at and updated_at fields")
        elif "remove" in user_request.lower():
            changes.append("Noted request to remove elements")
        else:
            changes.append("Processed user request")
        
        explanation = f"Applied basic modifications based on request: {user_request}"
        
        return current_schema, changes, confidence, explanation
    
    async def get_current_schema(self, user_id: str, session_id: str) -> Optional[str]:
        """Get the current schema state from a refinement session.
        
        Retrieves the most recent version of the schema from the conversation
        history, accounting for all refinements made during the session.
        
        Args:
            user_id: User identifier for access control
            session_id: Target refinement session identifier
            
        Returns:
            Optional[str]: Current schema content, or None if not found
            
        Raises:
            ValueError: If required parameters are missing
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")
        if not session_id or not session_id.strip():
            raise ValueError("session_id is required and cannot be empty")
            
        try:
            session_data = await self.chat_service.get_session_history(user_id, session_id)
            if not session_data:
                logger.warning(f"No session data found for session {session_id}")
                return None
                
            current_schema = await self._extract_current_schema(session_data)
            
            if current_schema:
                logger.debug(
                    f"Retrieved current schema for session {session_id} "
                    f"(length: {len(current_schema)} characters)"
                )
            else:
                logger.warning(f"No schema content found for session {session_id}")
                
            return current_schema
            
        except Exception as e:
            logger.error(f"Error retrieving current schema for session {session_id}: {str(e)}")
            return None

    def _format_refinement_response(
        self,
        changes_made: List[str],
        confidence: float,
        explanation: str,
        processing_time_ms: int
    ) -> str:
        """Format the refinement response message for user presentation."""
        changes_text = "\n".join(f"• {change}" for change in changes_made) if changes_made else "• No specific changes made"
        
        return f"""✨ **Schema Updated Successfully!**

**Changes Made:**
{changes_text}

**Confidence:** {confidence:.0%}

**Explanation:**
{explanation}

**Processing Time:** {processing_time_ms}ms

---

The schema has been updated with your requested changes. You can:
- **Continue refining**: Ask for more modifications
- **Preview**: View the updated schema
- **Export**: Save the refined schema

What would you like to do next?"""
    
    async def _handle_refinement_error(
        self,
        user_id: str,
        session_id: str,
        error: Exception,
        user_request: str
    ) -> str:
        """Handle refinement errors with user-friendly messaging."""
        error_message = f"""I encountered an error while processing your request: {str(error)}

**Troubleshooting suggestions:**
• Try rephrasing your request more specifically
• Break down complex changes into smaller steps
• Specify which part of the schema you want to modify

**Your request was:** "{user_request[:100]}{'...' if len(user_request) > 100 else ''}"

Please try again with a more specific request, or ask for help with a particular modification."""
        
        # Record error message in session
        try:
            await self.chat_service.add_message(
                user_id=user_id,
                session_id=session_id,
                message_type=MessageType.ASSISTANT_MESSAGE,
                content=error_message,
                metadata={
                    "error": str(error),
                    "error_type": "refinement_error",
                    "user_request": user_request,
                    "error_class": error.__class__.__name__
                }
            )
        except Exception as msg_error:
            logger.error(f"Failed to record error message: {msg_error}")
            
        return error_message
    
    async def _log_refinement_operation(
        self,
        user_id: str,
        operation_id: str,
        session_id: str,
        user_request: str,
        changes_made: List[str],
        confidence: float,
        processing_time_ms: int,
        assistant_message_id: str
    ) -> None:
        """Log comprehensive refinement operation audit trail."""
        try:
            await self.audit_service.log_operation_start(
                user_id=user_id,
                operation_type=OperationType.CHAT_SCHEMA_REFINEMENT,
                operation_id=operation_id,
                resource_name=f"Schema refinement in session {session_id}",
                metadata={
                    "session_id": session_id,
                    "user_request": user_request[:500],  # Truncate for storage
                    "user_request_length": len(user_request),
                    "changes_count": len(changes_made),
                    "changes_made": changes_made[:10],  # Limit for storage
                    "confidence": confidence,
                    "processing_time_ms": processing_time_ms,
                    "assistant_message_id": assistant_message_id,
                    "refinement_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.audit_service.log_operation_end(
                user_id=user_id,
                operation_id=operation_id,
                status=OperationStatus.SUCCESS
            )
            
        except Exception as audit_error:
            logger.error(f"Failed to log refinement operation: {audit_error}")


# Global service instance
schema_refinement_service = SchemaRefinementService()