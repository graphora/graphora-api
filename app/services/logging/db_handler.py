"""Database handler for logging merge operations"""
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from app.config import settings
from app.services.storage.neo4j import Neo4jStorage

class DatabaseHandler(logging.Handler):
    """Logging handler that stores logs in a database"""
    
    def __init__(self, level=logging.NOTSET):
        """Initialize the handler with Neo4j connection"""
        super().__init__(level)
        self.storage = None
        self.initialized = False
        
    async def _initialize(self):
        """Initialize Neo4j connection lazily"""
        if not self.initialized:
            self.storage = Neo4jStorage(
                uri=settings.NEO4J_URI,
                username=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                database=settings.NEO4J_DB
            )
            self.initialized = True
    
    def emit(self, record):
        """Process a log record by storing it in the database"""
        try:
            # Extract log data
            log_entry = self._format_log_entry(record)
            
            # Store log asynchronously
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a task to store the log
                asyncio.create_task(self._store_log(log_entry))
            else:
                # Run the coroutine directly
                loop.run_until_complete(self._store_log(log_entry))
                
        except Exception as e:
            # Don't raise exceptions in logging handlers
            self.handleError(record)
            print(f"Error in DatabaseHandler: {str(e)}")
    
    def _format_log_entry(self, record) -> Dict[str, Any]:
        """Format log record into a database-friendly structure"""
        # Parse the message as JSON if possible
        try:
            message_data = json.loads(record.msg)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, use the formatted message
            message_data = {"message": self.format(record)}
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "merge_id": message_data.get("merge_id", "unknown"),
            "operation_id": message_data.get("operation_id", "unknown"),
            "action": message_data.get("action", "unknown"),
            "status": message_data.get("status", "unknown"),
            "user_id": message_data.get("user_id"),
            "details": message_data if isinstance(message_data, dict) else {"message": message_data},
            "source": f"{record.pathname}:{record.lineno}",
            "process_id": record.process,
            "thread_id": record.thread
        }
        
        return log_entry
    
    async def _store_log(self, log_entry: Dict[str, Any]):
        """Store log entry in the database"""
        await self._initialize()
        
        # Create Cypher query to store log
        query = """
        CREATE (l:MergeLog {
            id: randomUUID(),
            timestamp: $timestamp,
            level: $level,
            logger: $logger,
            merge_id: $merge_id,
            operation_id: $operation_id,
            action: $action,
            status: $status,
            user_id: $user_id,
            details: $details,
            source: $source,
            process_id: $process_id,
            thread_id: $thread_id
        })
        """
        
        # Execute query
        try:
            await self.storage.execute_query(query, log_entry)
        except Exception as e:
            print(f"Failed to store log in database: {str(e)}") 