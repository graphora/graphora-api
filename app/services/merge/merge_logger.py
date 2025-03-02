"""Service for logging merge activities"""
from typing import Dict, List, Any, Optional
import json
import logging
import uuid
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from app.config import settings
from pathlib import Path
import asyncio

class MergeLogger:
    """Service for logging merge activities"""
    
    def __init__(self, merge_id: str, user_id: Optional[str] = None):
        """Initialize logger with merge context"""
        self.merge_id = merge_id
        self.user_id = user_id
        self.operation_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"merge.{merge_id}")
        self._configure_logger()
        
        # Log merge initialization
        self.log_merge_start()
    
    def _configure_logger(self):
        """Configure logger with appropriate handlers"""
        # Set log level from settings
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        
        # Ensure log directory exists
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler for this specific merge
        file_path = log_dir / f"merge_{self.merge_id}.log"
        file_handler = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=settings.LOG_MAX_FILE_SIZE_MB * 1024 * 1024,
            backupCount=settings.LOG_BACKUP_COUNT
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Add JSON handler for database logging
        if settings.LOG_TO_DATABASE:
            from app.services.logging.db_handler import DatabaseHandler
            db_handler = DatabaseHandler()
            self.logger.addHandler(db_handler)
    
    def _create_log_entry(self, 
                         action: str, 
                         status: str, 
                         details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized log entry dictionary"""
        entry = {
            "merge_id": self.merge_id,
            "operation_id": self.operation_id,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "status": status,
            "user_id": self.user_id
        }
        
        if details:
            entry["details"] = details
            
        return entry
    
    def log_merge_start(self):
        """Log merge process initialization"""
        entry = self._create_log_entry(
            action="merge_start",
            status="initialized",
            details={
                "source": "staging",
                "destination": "production",
                "initiated_at": datetime.now().isoformat()
            }
        )
        self.logger.info(json.dumps(entry))
        return entry
    
    def log_validation(self, 
                      status: str, 
                      validation_results: Dict[str, Any]):
        """Log pre-merge validation results"""
        # Handle coroutines in validation_results
        if asyncio.iscoroutine(validation_results):
            validation_results = {"<coroutine>": "validation_results"}
        else:
            # Sanitize validation_results to remove any coroutines
            sanitized_results = {}
            for key, value in validation_results.items():
                if asyncio.iscoroutine(value):
                    sanitized_results[key] = f"<coroutine {key}>"
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    sanitized_nested = {}
                    for k, v in value.items():
                        if asyncio.iscoroutine(v):
                            sanitized_nested[k] = f"<coroutine {k}>"
                        else:
                            sanitized_nested[k] = v
                    sanitized_results[key] = sanitized_nested
                else:
                    sanitized_results[key] = value
            validation_results = sanitized_results
            
        entry = self._create_log_entry(
            action="validation",
            status=status,
            details={
                "validation_results": validation_results,
                "passed": status == "success"
            }
        )
        self.logger.info(json.dumps(entry))
        return entry
    
    def log_merge_execution(self, 
                           node_count: int, 
                           relationship_count: int, 
                           status: str):
        """Log merge execution"""
        # Handle coroutines
        if asyncio.iscoroutine(node_count):
            node_count = "<coroutine node_count>"
        if asyncio.iscoroutine(relationship_count):
            relationship_count = "<coroutine relationship_count>"
            
        entry = self._create_log_entry(
            action="merge_execution",
            status=status,
            details={
                "node_count": node_count,
                "relationship_count": relationship_count,
                "execution_time": datetime.now().isoformat()
            }
        )
        self.logger.info(json.dumps(entry))
        return entry
    
    def log_entity_merge(self, 
                        entity_type: str, 
                        entity_id: str, 
                        status: str, 
                        properties: Dict[str, Any]):
        """Log individual entity merge operation"""
        # Handle coroutines
        if asyncio.iscoroutine(entity_type):
            entity_type = "<coroutine entity_type>"
        if asyncio.iscoroutine(entity_id):
            entity_id = "<coroutine entity_id>"
        if asyncio.iscoroutine(properties):
            properties = {"<coroutine>": "properties"}
        else:
            # Sanitize properties to remove any coroutines
            sanitized_properties = {}
            for key, value in properties.items():
                if asyncio.iscoroutine(value):
                    sanitized_properties[key] = f"<coroutine {key}>"
                else:
                    sanitized_properties[key] = value
            properties = sanitized_properties
            
        entry = self._create_log_entry(
            action="entity_merge",
            status=status,
            details={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "properties": properties
            }
        )
        self.logger.debug(json.dumps(entry))
        return entry
    
    def log_relationship_merge(self, 
                             source_id: str, 
                             target_id: str, 
                             rel_type: str, 
                             status: str):
        """Log individual relationship merge operation"""
        # Handle coroutines
        if asyncio.iscoroutine(source_id):
            source_id = "<coroutine source_id>"
        if asyncio.iscoroutine(target_id):
            target_id = "<coroutine target_id>"
        if asyncio.iscoroutine(rel_type):
            rel_type = "<coroutine rel_type>"
            
        entry = self._create_log_entry(
            action="relationship_merge",
            status=status,
            details={
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": rel_type
            }
        )
        self.logger.debug(json.dumps(entry))
        return entry
    
    def log_error(self, 
                 error_type: str, 
                 error_message: str, 
                 context: Dict[str, Any]):
        """Log error during merge process"""
        # Handle coroutines in context
        sanitized_context = {}
        for key, value in context.items():
            if asyncio.iscoroutine(value):
                # Skip coroutines as they can't be serialized
                sanitized_context[key] = f"<coroutine {key}>"
            else:
                sanitized_context[key] = value
                
        entry = self._create_log_entry(
            action="error",
            status="failed",
            details={
                "error_type": error_type,
                "error_message": error_message,
                "context": sanitized_context
            }
        )
        self.logger.error(json.dumps(entry))
        return entry
    
    def log_rollback(self, 
                    reason: str, 
                    affected_entities: List[str], 
                    status: str):
        """Log rollback operation"""
        # Handle coroutines
        if asyncio.iscoroutine(reason):
            reason = "<coroutine reason>"
        if asyncio.iscoroutine(affected_entities):
            affected_entities = ["<coroutine affected_entities>"]
        else:
            # Sanitize affected_entities to remove any coroutines
            sanitized_entities = []
            for entity in affected_entities:
                if asyncio.iscoroutine(entity):
                    sanitized_entities.append("<coroutine entity>")
                else:
                    sanitized_entities.append(entity)
            affected_entities = sanitized_entities
        if asyncio.iscoroutine(status):
            status = "<coroutine status>"
            
        entry = self._create_log_entry(
            action="rollback",
            status=status,
            details={
                "reason": reason,
                "affected_entities": affected_entities,
                "rollback_time": datetime.now().isoformat()
            }
        )
        self.logger.warning(json.dumps(entry))
        return entry
    
    def log_merge_completion(self, 
                            status: str, 
                            metrics: Dict[str, Any]):
        """Log merge completion"""
        # Handle coroutines in metrics
        if asyncio.iscoroutine(metrics):
            metrics = {"<coroutine>": "metrics"}
        else:
            # Sanitize metrics to remove any coroutines
            sanitized_metrics = {}
            for key, value in metrics.items():
                if asyncio.iscoroutine(value):
                    sanitized_metrics[key] = f"<coroutine {key}>"
                else:
                    sanitized_metrics[key] = value
            metrics = sanitized_metrics
            
        entry = self._create_log_entry(
            action="merge_completion",
            status=status,
            details={
                "completion_time": datetime.now().isoformat(),
                "metrics": metrics
            }
        )
        self.logger.info(json.dumps(entry))
        return entry
    
    @classmethod
    def get_logs_for_merge(cls, merge_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve logs for a specific merge operation"""
        log_path = Path(settings.LOG_DIR) / f"merge_{merge_id}.log"
        if not log_path.exists():
            return []
        
        logs = []
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    # Extract the JSON part from the log line
                    json_start = line.find('{')
                    if json_start != -1:
                        json_str = line[json_start:]
                        log_entry = json.loads(json_str)
                        logs.append(log_entry)
                        if len(logs) >= limit:
                            break
                except json.JSONDecodeError:
                    continue
        
        return logs
    
    @classmethod
    async def get_logs_from_database(cls, merge_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve logs from database for a specific merge operation"""
        from app.services.storage.neo4j import Neo4jStorage
        
        storage = Neo4jStorage(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DB
        )
        
        query = """
        MATCH (l:MergeLog)
        WHERE l.merge_id = $merge_id
        RETURN l
        ORDER BY l.timestamp DESC
        LIMIT $limit
        """
        
        params = {
            "merge_id": merge_id,
            "limit": limit
        }
        
        result = await storage.execute_query(query, params)
        logs = []
        
        for record in result:
            log_node = record.get("l")
            if log_node:
                log_entry = dict(log_node)
                logs.append(log_entry)
        
        return logs 