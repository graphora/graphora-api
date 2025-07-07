"""Audit Trail Service for tracking operations in Supabase"""
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger(__name__)

class OperationType(str, Enum):
    """Types of operations to audit"""
    ONTOLOGY_STORED = "ontology_stored"
    TRANSFORM_STARTED = "transform_started"
    TRANSFORM_COMPLETED = "transform_completed"
    MERGE_STARTED = "merge_started"
    MERGE_COMPLETED = "merge_completed"
    SCHEMA_GENERATION = "schema_generation"
    SCHEMA_SEARCH = "schema_search"
    SCHEMA_REFINEMENT = "schema_refinement"
    SCHEMA_CREATE = "schema_create"
    
    # Chat-related operations
    CHAT_SESSION_STARTED = "chat_session_started"
    CHAT_MESSAGE_SENT = "chat_message_sent"
    CHAT_MESSAGE_RECEIVED = "chat_message_received"
    CHAT_SESSION_ENDED = "chat_session_ended"
    CHAT_SCHEMA_REFINEMENT = "chat_schema_refinement"

class OperationStatus(str, Enum):
    """Status of operations"""
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"

class AuditService:
    """Service for logging audit trails"""
    
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")
        
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
    
    async def log_operation_start(
        self,
        user_id: str,
        operation_type: OperationType,
        operation_id: str,
        resource_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log the start of an operation
        
        Returns:
            audit_id: ID of the created audit record
        """
        try:
            audit_data = {
                "user_id": user_id,
                "operation_type": operation_type.value,
                "operation_id": operation_id,
                "resource_name": resource_name,
                "status": OperationStatus.IN_PROGRESS.value,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("audit_trail").insert(audit_data).execute()
            
            if result.data and len(result.data) > 0:
                audit_id = result.data[0]["id"]
                logger.info(f"Started audit trail {audit_id} for {operation_type.value} by user {user_id}")
                return audit_id
            else:
                logger.error(f"Failed to create audit trail for {operation_type.value}")
                return ""
                
        except Exception as e:
            logger.error(f"Error logging operation start: {str(e)}")
            return ""
    
    async def log_operation_success(
        self,
        audit_id: str,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log successful completion of an operation"""
        try:
            update_data = {
                "status": OperationStatus.SUCCESS.value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if duration_ms is not None:
                update_data["duration_ms"] = duration_ms
            
            if metadata:
                # Merge with existing metadata
                existing_record = self.supabase.table("audit_trail").select("metadata").eq("id", audit_id).execute()
                if existing_record.data and len(existing_record.data) > 0:
                    existing_metadata = existing_record.data[0].get("metadata", {})
                    update_data["metadata"] = {**existing_metadata, **metadata}
                else:
                    update_data["metadata"] = metadata
            
            result = self.supabase.table("audit_trail").update(update_data).eq("id", audit_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully completed audit trail {audit_id}")
                return True
            else:
                logger.error(f"Failed to update audit trail {audit_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging operation success: {str(e)}")
            return False
    
    async def log_operation_failure(
        self,
        audit_id: str,
        error_message: str,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log failed completion of an operation"""
        try:
            update_data = {
                "status": OperationStatus.FAILED.value,
                "error_message": error_message,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if duration_ms is not None:
                update_data["duration_ms"] = duration_ms
            
            if metadata:
                # Merge with existing metadata
                existing_record = self.supabase.table("audit_trail").select("metadata").eq("id", audit_id).execute()
                if existing_record.data and len(existing_record.data) > 0:
                    existing_metadata = existing_record.data[0].get("metadata", {})
                    update_data["metadata"] = {**existing_metadata, **metadata}
                else:
                    update_data["metadata"] = metadata
            
            result = self.supabase.table("audit_trail").update(update_data).eq("id", audit_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Logged failure for audit trail {audit_id}: {error_message}")
                return True
            else:
                logger.error(f"Failed to update audit trail {audit_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging operation failure: {str(e)}")
            return False
    
    async def log_direct_operation(
        self,
        user_id: str,
        operation_type: OperationType,
        operation_id: str,
        status: OperationStatus,
        resource_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> str:
        """Log a complete operation in one call"""
        try:
            audit_data = {
                "user_id": user_id,
                "operation_type": operation_type.value,
                "operation_id": operation_id,
                "resource_name": resource_name,
                "status": status.value,
                "metadata": metadata or {},
                "error_message": error_message,
                "duration_ms": duration_ms,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("audit_trail").insert(audit_data).execute()
            
            if result.data and len(result.data) > 0:
                audit_id = result.data[0]["id"]
                logger.info(f"Logged audit trail {audit_id} for {operation_type.value} by user {user_id}")
                return audit_id
            else:
                logger.error(f"Failed to create audit trail for {operation_type.value}")
                return ""
                
        except Exception as e:
            logger.error(f"Error logging direct operation: {str(e)}")
            return ""
    
    async def get_user_audit_trail(
        self,
        user_id: str,
        operation_type: Optional[OperationType] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get audit trail for a user"""
        try:
            query = self.supabase.table("audit_trail").select("*").eq("user_id", user_id)
            
            if operation_type:
                query = query.eq("operation_type", operation_type.value)
            
            result = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Error fetching audit trail: {str(e)}")
            return []
    
    async def get_audit_summary(self, user_id: str) -> Dict[str, Any]:
        """Get audit trail summary for dashboard"""
        try:
            # Get counts by operation type
            result = self.supabase.table("audit_trail").select("operation_type, status").eq("user_id", user_id).execute()
            
            operations = result.data or []
            
            summary = {
                "total_operations": len(operations),
                "by_type": {},
                "by_status": {},
                "recent_operations": []
            }
            
            # Count by operation type (only successful operations) and status (all operations)
            for op in operations:
                op_type = op.get("operation_type", "unknown")
                status = op.get("status", "unknown")
                
                # For by_type, only count successful operations (for dashboard metrics)
                if status == "success":
                    if op_type not in summary["by_type"]:
                        summary["by_type"][op_type] = 0
                    summary["by_type"][op_type] += 1
                
                # For by_status, count all operations
                if status not in summary["by_status"]:
                    summary["by_status"][status] = 0
                summary["by_status"][status] += 1
            
            # Get recent operations
            recent_result = self.supabase.table("audit_trail").select(
                "*"
            ).eq("user_id", user_id).order("created_at", desc=True).limit(10).execute()
            
            summary["recent_operations"] = recent_result.data or []
            
            return summary
            
        except Exception as e:
            logger.error(f"Error fetching audit summary: {str(e)}")
            return {
                "total_operations": 0,
                "by_type": {},
                "by_status": {},
                "recent_operations": []
            }

# Create global instance
audit_service = AuditService() 