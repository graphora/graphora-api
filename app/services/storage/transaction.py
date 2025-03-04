"""Transaction management for graph storage operations"""
import uuid
import logging
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from datetime import datetime
from abc import ABC, abstractmethod
import threading
import inspect

from neo4j import AsyncSession, Transaction

from app.services.storage.exceptions import StorageError

# Configure logger
logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')

# Thread-local storage for the current transaction
_transaction_context = threading.local()

class TransactionManager(ABC):
    """Manage database transactions for merge operations"""
    
    @abstractmethod
    async def begin_transaction(self, merge_id: str) -> str:
        """Begin a new transaction and return transaction ID
        
        Args:
            merge_id: ID of the merge operation
            
        Returns:
            str: Transaction ID
        """
        pass
        
    @abstractmethod
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit the transaction identified by transaction_id
        
        Args:
            transaction_id: ID of the transaction to commit
            
        Returns:
            bool: True if commit was successful, False otherwise
        """
        pass
        
    @abstractmethod
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback the transaction identified by transaction_id
        
        Args:
            transaction_id: ID of the transaction to rollback
            
        Returns:
            bool: True if rollback was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction data by ID
        
        Args:
            transaction_id: ID of the transaction to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Transaction data if found, None otherwise
        """
        pass
    
    async def start_transaction(self):
        """Start a transaction that can be used as an async context manager
        
        Usage:
            async with transaction_manager.start_transaction() as tx:
                # Do operations with tx
                
        Returns:
            AsyncContextManager: Context manager for the transaction
        """
        class TransactionContext:
            def __init__(self, manager):
                self.manager = manager
                self.transaction_id = None
                
            async def __aenter__(self):
                # Generate a unique merge ID for this transaction
                merge_id = f"tx_{uuid.uuid4().hex}"
                self.transaction_id = await self.manager.begin_transaction(merge_id)
                return self.transaction_id
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    # Exception occurred, rollback
                    await self.manager.rollback_transaction(self.transaction_id)
                else:
                    # No exception, commit
                    await self.manager.commit_transaction(self.transaction_id)
                    
        return TransactionContext(self)
        
    async def execute_in_transaction(self, callback: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute the callback within a transaction
        
        Args:
            callback: Async function to execute within transaction
            *args: Arguments to pass to callback
            **kwargs: Keyword arguments to pass to callback
            
        Returns:
            Any: Result of callback execution
            
        Raises:
            Exception: Any exception raised by callback
        """
        merge_id = kwargs.get('merge_id', str(uuid.uuid4()))
        transaction_id = await self.begin_transaction(merge_id)
        
        try:
            # Add transaction_id to kwargs
            kwargs['transaction_id'] = transaction_id
            
            # Execute callback
            result = await callback(*args, **kwargs)
            
            # Commit transaction
            success = await self.commit_transaction(transaction_id)
            if not success:
                logger.error(f"Failed to commit transaction {transaction_id}")
                raise Exception(f"Failed to commit transaction {transaction_id}")
                
            return result
        except Exception as e:
            # Rollback transaction on error
            logger.error(f"Error in transaction {transaction_id}: {str(e)}")
            await self.rollback_transaction(transaction_id)
            raise e


class Neo4jTransactionManager(TransactionManager):
    """Neo4j implementation of transaction management"""
    
    def __init__(self, driver, timeout: int = 300):
        """Initialize Neo4j transaction manager
        
        Args:
            driver: Neo4j driver instance
            timeout: Transaction timeout in seconds (default: 300)
        """
        self.driver = driver
        self.timeout = timeout
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        
    async def begin_transaction(self, merge_id: str) -> str:
        """Begin a new Neo4j transaction
        
        Args:
            merge_id: ID of the merge operation
            
        Returns:
            str: Transaction ID
        """
        transaction_id = f"tx_{merge_id}_{uuid.uuid4().hex}"
        
        # Create session and transaction
        session_result = self.driver.session(database="neo4j")
        
        # Handle both synchronous and asynchronous session creation
        if inspect.isawaitable(session_result):
            session = await session_result
        else:
            session = session_result
            
        # Begin transaction
        tx_result = session.begin_transaction(timeout=self.timeout)
        
        # Handle both synchronous and asynchronous transaction creation
        if inspect.isawaitable(tx_result):
            tx = await tx_result
        else:
            tx = tx_result
        
        # Store transaction data
        self.active_transactions[transaction_id] = {
            "session": session,
            "tx": tx,
            "start_time": datetime.now(),
            "merge_id": merge_id,
            "metadata": {
                "transaction_id": transaction_id,
                "merge_id": merge_id,
                "start_time": datetime.now().isoformat()
            }
        }
        
        # Log transaction start
        logger.info(f"Started transaction {transaction_id} for merge {merge_id}")
        
        # Store transaction in thread-local storage
        _transaction_context.current_transaction = transaction_id
        
        return transaction_id
        
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit Neo4j transaction
        
        Args:
            transaction_id: ID of the transaction to commit
            
        Returns:
            bool: True if commit was successful, False otherwise
        """
        if transaction_id not in self.active_transactions:
            logger.error(f"Transaction {transaction_id} not found")
            return False
            
        tx_data = self.active_transactions[transaction_id]
        
        try:
            # Commit transaction
            await tx_data["tx"].commit()
            await tx_data["session"].close()
            
            # Calculate duration
            duration = (datetime.now() - tx_data["start_time"]).total_seconds()
            
            # Update metadata
            tx_data["metadata"]["end_time"] = datetime.now().isoformat()
            tx_data["metadata"]["duration_seconds"] = duration
            tx_data["metadata"]["status"] = "committed"
            
            # Log transaction commit
            logger.info(f"Committed transaction {transaction_id} after {duration:.2f}s")
            
            # Remove transaction from active transactions
            del self.active_transactions[transaction_id]
            
            # Clear thread-local storage
            if hasattr(_transaction_context, 'current_transaction'):
                delattr(_transaction_context, 'current_transaction')
            
            return True
            
        except Exception as e:
            logger.error(f"Error committing transaction {transaction_id}: {str(e)}")
            await self.rollback_transaction(transaction_id)
            return False
            
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback Neo4j transaction
        
        Args:
            transaction_id: ID of the transaction to rollback
            
        Returns:
            bool: True if rollback was successful, False otherwise
        """
        if transaction_id not in self.active_transactions:
            logger.error(f"Transaction {transaction_id} not found")
            return False
            
        tx_data = self.active_transactions[transaction_id]
        
        try:
            # Rollback transaction
            await tx_data["tx"].rollback()
            await tx_data["session"].close()
            
            # Calculate duration
            duration = (datetime.now() - tx_data["start_time"]).total_seconds()
            
            # Update metadata
            tx_data["metadata"]["end_time"] = datetime.now().isoformat()
            tx_data["metadata"]["duration_seconds"] = duration
            tx_data["metadata"]["status"] = "rolled_back"
            
            # Log transaction rollback
            logger.warning(f"Rolled back transaction {transaction_id} after {duration:.2f}s")
            
            # Remove transaction from active transactions
            del self.active_transactions[transaction_id]
            
            # Clear thread-local storage
            if hasattr(_transaction_context, 'current_transaction'):
                delattr(_transaction_context, 'current_transaction')
            
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back transaction {transaction_id}: {str(e)}")
            return False
    
    def get_transaction_metadata(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a transaction
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            Optional[Dict[str, Any]]: Transaction metadata or None if not found
        """
        if transaction_id in self.active_transactions:
            return self.active_transactions[transaction_id]["metadata"]
        return None
    
    def get_active_transaction_count(self) -> int:
        """Get count of active transactions
        
        Returns:
            int: Number of active transactions
        """
        return len(self.active_transactions)
    
    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction data by ID
        
        Args:
            transaction_id: ID of the transaction to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Transaction data if found, None otherwise
        """
        return self.active_transactions.get(transaction_id)
    
    def get_session(self, transaction_id: str) -> Optional[AsyncSession]:
        """Get Neo4j session object
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            Optional[AsyncSession]: Neo4j session object or None if not found
        """
        if transaction_id in self.active_transactions:
            return self.active_transactions[transaction_id]["session"]
        return None
    
    def get_current_transaction(self) -> Optional[Dict]:
        """Get the current transaction from thread-local storage
        
        Returns:
            Optional[Dict]: Transaction data if available, None otherwise
        """
        if hasattr(_transaction_context, 'current_transaction'):
            transaction_id = _transaction_context.current_transaction
            if transaction_id in self.active_transactions:
                return self.active_transactions[transaction_id]
        return None
    
    async def start_transaction(self):
        """Start a Neo4j transaction that can be used as an async context manager
        
        Usage:
            async with transaction_manager.start_transaction() as tx:
                # Do operations with tx
                
        Returns:
            AsyncContextManager: Context manager for the Neo4j transaction
        """
        class Neo4jTransactionContext:
            def __init__(self, manager):
                self.manager = manager
                self.transaction_id = None
                
            async def __aenter__(self):
                # Generate a unique merge ID for this transaction
                merge_id = f"tx_{uuid.uuid4().hex}"
                self.transaction_id = await self.manager.begin_transaction(merge_id)
                # Return the actual transaction object from Neo4j
                tx_data = self.manager.active_transactions.get(self.transaction_id, {})
                return tx_data.get("tx")
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    # Exception occurred, rollback
                    await self.manager.rollback_transaction(self.transaction_id)
                else:
                    # No exception, commit
                    await self.manager.commit_transaction(self.transaction_id)
                    
        return Neo4jTransactionContext(self) 