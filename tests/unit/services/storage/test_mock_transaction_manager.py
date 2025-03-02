"""Mock implementation of TransactionManager for testing"""
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.services.storage.transaction import TransactionManager


class MockTransactionManager(TransactionManager):
    """Mock implementation of TransactionManager for testing"""
    
    def __init__(self):
        """Initialize mock transaction manager"""
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        self.transaction_history: List[Dict[str, Any]] = []
        self.begin_called = False
        self.commit_called = False
        self.rollback_called = False
        self.last_transaction_id = None
        
    async def begin_transaction(self, merge_id: str) -> str:
        """Begin a mock transaction
        
        Args:
            merge_id: ID of the merge operation
            
        Returns:
            str: Transaction ID
        """
        self.begin_called = True
        transaction_id = f"mock_tx_{merge_id}_{uuid.uuid4().hex}"
        self.last_transaction_id = transaction_id
        
        # Store transaction data
        self.active_transactions[transaction_id] = {
            "merge_id": merge_id,
            "start_time": datetime.now(),
            "status": "active",
            "operations": []
        }
        
        # Add to history
        self.transaction_history.append({
            "transaction_id": transaction_id,
            "merge_id": merge_id,
            "action": "begin",
            "timestamp": datetime.now().isoformat()
        })
        
        return transaction_id
        
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a mock transaction
        
        Args:
            transaction_id: ID of the transaction to commit
            
        Returns:
            bool: True if commit was successful, False otherwise
        """
        self.commit_called = True
        
        if transaction_id not in self.active_transactions:
            return False
        
        # Update transaction data
        self.active_transactions[transaction_id]["status"] = "committed"
        self.active_transactions[transaction_id]["end_time"] = datetime.now()
        
        # Add to history
        self.transaction_history.append({
            "transaction_id": transaction_id,
            "merge_id": self.active_transactions[transaction_id]["merge_id"],
            "action": "commit",
            "timestamp": datetime.now().isoformat()
        })
        
        # Remove from active transactions
        tx_data = self.active_transactions[transaction_id]
        del self.active_transactions[transaction_id]
        
        return True
        
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback a mock transaction
        
        Args:
            transaction_id: ID of the transaction to rollback
            
        Returns:
            bool: True if rollback was successful, False otherwise
        """
        self.rollback_called = True
        
        if transaction_id not in self.active_transactions:
            return False
        
        # Update transaction data
        self.active_transactions[transaction_id]["status"] = "rolled_back"
        self.active_transactions[transaction_id]["end_time"] = datetime.now()
        
        # Add to history
        self.transaction_history.append({
            "transaction_id": transaction_id,
            "merge_id": self.active_transactions[transaction_id]["merge_id"],
            "action": "rollback",
            "timestamp": datetime.now().isoformat()
        })
        
        # Remove from active transactions
        tx_data = self.active_transactions[transaction_id]
        del self.active_transactions[transaction_id]
        
        return True
    
    def record_operation(self, transaction_id: str, operation: str, data: Dict[str, Any]) -> None:
        """Record an operation within a transaction
        
        Args:
            transaction_id: ID of the transaction
            operation: Type of operation (e.g., 'create_node', 'update_node', etc.)
            data: Operation data
        """
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]["operations"].append({
                "operation": operation,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_transaction_history(self) -> List[Dict[str, Any]]:
        """Get transaction history
        
        Returns:
            List[Dict[str, Any]]: List of transaction history entries
        """
        return self.transaction_history
    
    def get_transaction_operations(self, transaction_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get operations for a transaction
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of operations or None if transaction not found
        """
        if transaction_id in self.active_transactions:
            return self.active_transactions[transaction_id]["operations"]
        
        # Check in history for completed transactions
        for entry in self.transaction_history:
            if entry["transaction_id"] == transaction_id and entry["action"] == "begin":
                # Find the corresponding commit/rollback entry
                for end_entry in self.transaction_history:
                    if (end_entry["transaction_id"] == transaction_id and 
                        end_entry["action"] in ["commit", "rollback"]):
                        # This transaction was completed
                        return []
        
        return None
    
    def reset(self) -> None:
        """Reset the mock transaction manager"""
        self.active_transactions = {}
        self.transaction_history = []
        self.begin_called = False
        self.commit_called = False
        self.rollback_called = False
        self.last_transaction_id = None 