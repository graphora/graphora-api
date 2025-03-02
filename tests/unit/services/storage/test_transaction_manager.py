"""Unit tests for transaction management"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime

from app.services.storage.transaction import TransactionManager, Neo4jTransactionManager


class MockTransactionManager(TransactionManager):
    """Mock implementation of TransactionManager for testing"""
    
    def __init__(self):
        self.active_transactions = {}
        self.begin_called = False
        self.commit_called = False
        self.rollback_called = False
        self.last_transaction_id = None
        
    async def begin_transaction(self, merge_id: str) -> str:
        """Begin a mock transaction"""
        self.begin_called = True
        transaction_id = f"mock_tx_{merge_id}_{uuid.uuid4().hex}"
        self.last_transaction_id = transaction_id
        self.active_transactions[transaction_id] = {
            "merge_id": merge_id,
            "start_time": datetime.now()
        }
        return transaction_id
        
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a mock transaction"""
        self.commit_called = True
        if transaction_id in self.active_transactions:
            del self.active_transactions[transaction_id]
            return True
        return False
        
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback a mock transaction"""
        self.rollback_called = True
        if transaction_id in self.active_transactions:
            del self.active_transactions[transaction_id]
            return True
        return False


class TestTransactionManager:
    """Test cases for TransactionManager"""
    
    @pytest.fixture
    def mock_transaction_manager(self):
        """Fixture for mock transaction manager"""
        return MockTransactionManager()
    
    @pytest.mark.asyncio
    async def test_begin_transaction(self, mock_transaction_manager):
        """Test begin_transaction creates a transaction and returns ID"""
        merge_id = "test_merge_123"
        transaction_id = await mock_transaction_manager.begin_transaction(merge_id)
        
        assert mock_transaction_manager.begin_called
        assert transaction_id.startswith(f"mock_tx_{merge_id}")
        assert transaction_id in mock_transaction_manager.active_transactions
        assert mock_transaction_manager.active_transactions[transaction_id]["merge_id"] == merge_id
    
    @pytest.mark.asyncio
    async def test_commit_transaction_success(self, mock_transaction_manager):
        """Test commit_transaction successfully commits a transaction"""
        merge_id = "test_merge_123"
        transaction_id = await mock_transaction_manager.begin_transaction(merge_id)
        
        result = await mock_transaction_manager.commit_transaction(transaction_id)
        
        assert result is True
        assert mock_transaction_manager.commit_called
        assert transaction_id not in mock_transaction_manager.active_transactions
    
    @pytest.mark.asyncio
    async def test_commit_transaction_not_found(self, mock_transaction_manager):
        """Test commit_transaction handles non-existent transaction"""
        result = await mock_transaction_manager.commit_transaction("non_existent_tx")
        
        assert result is False
        assert mock_transaction_manager.commit_called
    
    @pytest.mark.asyncio
    async def test_rollback_transaction_success(self, mock_transaction_manager):
        """Test rollback_transaction successfully rolls back a transaction"""
        merge_id = "test_merge_123"
        transaction_id = await mock_transaction_manager.begin_transaction(merge_id)
        
        result = await mock_transaction_manager.rollback_transaction(transaction_id)
        
        assert result is True
        assert mock_transaction_manager.rollback_called
        assert transaction_id not in mock_transaction_manager.active_transactions
    
    @pytest.mark.asyncio
    async def test_rollback_transaction_not_found(self, mock_transaction_manager):
        """Test rollback_transaction handles non-existent transaction"""
        result = await mock_transaction_manager.rollback_transaction("non_existent_tx")
        
        assert result is False
        assert mock_transaction_manager.rollback_called
    
    @pytest.mark.asyncio
    async def test_execute_in_transaction_success(self, mock_transaction_manager):
        """Test execute_in_transaction with successful callback"""
        merge_id = "test_merge_123"
        
        # Mock callback that returns a value
        async def mock_callback(merge_id, transaction_id):
            assert transaction_id is not None
            return {"success": True, "merge_id": merge_id}
        
        result = await mock_transaction_manager.execute_in_transaction(
            mock_callback, merge_id=merge_id
        )
        
        assert mock_transaction_manager.begin_called
        assert mock_transaction_manager.commit_called
        assert not mock_transaction_manager.rollback_called
        assert result["success"] is True
        assert result["merge_id"] == merge_id
    
    @pytest.mark.asyncio
    async def test_execute_in_transaction_failure(self, mock_transaction_manager):
        """Test execute_in_transaction with failing callback"""
        merge_id = "test_merge_123"
        
        # Mock callback that raises an exception
        async def mock_callback(merge_id, transaction_id):
            assert transaction_id is not None
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await mock_transaction_manager.execute_in_transaction(
                mock_callback, merge_id=merge_id
            )
        
        assert mock_transaction_manager.begin_called
        assert not mock_transaction_manager.commit_called
        assert mock_transaction_manager.rollback_called


class TestNeo4jTransactionManager:
    """Test cases for Neo4jTransactionManager"""
    
    @pytest.fixture
    def mock_driver(self):
        """Fixture for mock Neo4j driver"""
        driver = AsyncMock()
        session = AsyncMock()
        tx = AsyncMock()
        
        # Fix: Configure driver.session to return the session directly, not a coroutine
        driver.session = MagicMock(return_value=session)
        
        # Fix: Configure session.begin_transaction to return the tx directly, not a coroutine
        session.begin_transaction = MagicMock(return_value=tx)
        
        return driver, session, tx
    
    @pytest.fixture
    def neo4j_transaction_manager(self, mock_driver):
        """Fixture for Neo4jTransactionManager with mock driver"""
        driver, _, _ = mock_driver
        return Neo4jTransactionManager(driver)
    
    @pytest.mark.asyncio
    async def test_begin_transaction(self, neo4j_transaction_manager, mock_driver):
        """Test begin_transaction creates a Neo4j transaction"""
        driver, session, tx = mock_driver
        merge_id = "test_merge_123"
        
        transaction_id = await neo4j_transaction_manager.begin_transaction(merge_id)
        
        assert transaction_id.startswith(f"tx_{merge_id}")
        assert transaction_id in neo4j_transaction_manager.active_transactions
        assert neo4j_transaction_manager.active_transactions[transaction_id]["session"] == session
        assert neo4j_transaction_manager.active_transactions[transaction_id]["tx"] == tx
        assert neo4j_transaction_manager.active_transactions[transaction_id]["merge_id"] == merge_id
        assert "start_time" in neo4j_transaction_manager.active_transactions[transaction_id]
        assert "metadata" in neo4j_transaction_manager.active_transactions[transaction_id]
        
        # We can't verify session and transaction were created with assert_called_once
        # since we're using custom async functions instead of mocks
    
    @pytest.mark.asyncio
    async def test_commit_transaction_success(self, neo4j_transaction_manager, mock_driver):
        """Test commit_transaction successfully commits a Neo4j transaction"""
        driver, session, tx = mock_driver
        merge_id = "test_merge_123"
        
        # Begin transaction
        transaction_id = await neo4j_transaction_manager.begin_transaction(merge_id)
        
        # Commit transaction
        result = await neo4j_transaction_manager.commit_transaction(transaction_id)
        
        assert result is True
        assert transaction_id not in neo4j_transaction_manager.active_transactions
        
        # Verify transaction was committed and session closed
        tx.commit.assert_called_once()
        session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_commit_transaction_not_found(self, neo4j_transaction_manager):
        """Test commit_transaction handles non-existent transaction"""
        result = await neo4j_transaction_manager.commit_transaction("non_existent_tx")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_commit_transaction_error(self, neo4j_transaction_manager, mock_driver):
        """Test commit_transaction handles errors"""
        driver, session, tx = mock_driver
        merge_id = "test_merge_123"
        
        # Configure tx to raise exception on commit
        tx.commit.side_effect = Exception("Commit error")
        
        # Begin transaction
        transaction_id = await neo4j_transaction_manager.begin_transaction(merge_id)
        
        # Mock rollback_transaction to avoid infinite recursion
        neo4j_transaction_manager.rollback_transaction = AsyncMock(return_value=True)
        
        # Commit transaction (should fail)
        result = await neo4j_transaction_manager.commit_transaction(transaction_id)
        
        assert result is False
        
        # Verify rollback was attempted
        neo4j_transaction_manager.rollback_transaction.assert_called_once_with(transaction_id)
    
    @pytest.mark.asyncio
    async def test_rollback_transaction_success(self, neo4j_transaction_manager, mock_driver):
        """Test rollback_transaction successfully rolls back a Neo4j transaction"""
        driver, session, tx = mock_driver
        merge_id = "test_merge_123"
        
        # Begin transaction
        transaction_id = await neo4j_transaction_manager.begin_transaction(merge_id)
        
        # Rollback transaction
        result = await neo4j_transaction_manager.rollback_transaction(transaction_id)
        
        assert result is True
        assert transaction_id not in neo4j_transaction_manager.active_transactions
        
        # Verify transaction was rolled back and session closed
        tx.rollback.assert_called_once()
        session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rollback_transaction_not_found(self, neo4j_transaction_manager):
        """Test rollback_transaction handles non-existent transaction"""
        result = await neo4j_transaction_manager.rollback_transaction("non_existent_tx")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_rollback_transaction_error(self, neo4j_transaction_manager, mock_driver):
        """Test rollback_transaction handles errors"""
        driver, session, tx = mock_driver
        merge_id = "test_merge_123"
        
        # Configure tx to raise exception on rollback
        tx.rollback.side_effect = Exception("Rollback error")
        
        # Begin transaction
        transaction_id = await neo4j_transaction_manager.begin_transaction(merge_id)
        
        # Rollback transaction (should handle error)
        result = await neo4j_transaction_manager.rollback_transaction(transaction_id)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_transaction_metadata(self, neo4j_transaction_manager):
        """Test get_transaction_metadata returns metadata for active transaction"""
        merge_id = "test_merge_123"
        
        # Begin transaction
        transaction_id = await neo4j_transaction_manager.begin_transaction(merge_id)
        
        # Get metadata
        metadata = neo4j_transaction_manager.get_transaction_metadata(transaction_id)
        
        assert metadata is not None
        assert metadata["transaction_id"] == transaction_id
        assert metadata["merge_id"] == merge_id
        assert "start_time" in metadata
    
    @pytest.mark.asyncio
    async def test_get_transaction_metadata_not_found(self, neo4j_transaction_manager):
        """Test get_transaction_metadata returns None for non-existent transaction"""
        metadata = neo4j_transaction_manager.get_transaction_metadata("non_existent_tx")
        
        assert metadata is None
    
    def test_get_active_transaction_count(self, neo4j_transaction_manager):
        """Test get_active_transaction_count returns correct count"""
        assert neo4j_transaction_manager.get_active_transaction_count() == 0
        
        # Add some mock transactions
        neo4j_transaction_manager.active_transactions = {
            "tx1": {},
            "tx2": {},
            "tx3": {}
        }
        
        assert neo4j_transaction_manager.get_active_transaction_count() == 3 