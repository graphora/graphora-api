"""Unit tests for the DatabaseHandler"""
import json
import logging
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.logging.db_handler import DatabaseHandler


class TestDatabaseHandler:
    """Test cases for the DatabaseHandler class"""

    @pytest.fixture
    def mock_storage(self):
        """Mock Neo4jStorage for testing"""
        with patch('app.services.storage.neo4j.Neo4jStorage', autospec=True) as mock:
            mock_instance = mock.return_value
            mock_instance.execute_query = AsyncMock()
            yield mock_instance

    @pytest.fixture
    def db_handler(self):
        """Create a DatabaseHandler instance for testing"""
        handler = DatabaseHandler(level=logging.INFO)
        yield handler

    def test_initialization(self, db_handler):
        """Test that the handler initializes correctly"""
        assert db_handler.level == logging.INFO
        assert db_handler.storage is None
        assert db_handler.initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, db_handler, mock_storage):
        """Test lazy initialization of Neo4j connection"""
        with patch('app.services.storage.neo4j.Neo4jStorage', return_value=mock_storage):
            # Call _initialize
            await db_handler._initialize()
            
            # Check that storage was initialized
            assert db_handler.initialized is True
            assert db_handler.storage is not None

    def test_format_log_entry_json(self, db_handler):
        """Test formatting a log record with JSON message"""
        # Create a log record with JSON message
        json_data = {
            "merge_id": "test_merge_123",
            "operation_id": "test_op_456",
            "action": "validation",
            "status": "success",
            "user_id": "test_user",
            "details": {"valid": True}
        }
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path.py",
            lineno=123,
            msg=json.dumps(json_data),
            args=(),
            exc_info=None
        )
        
        # Format the record
        log_entry = db_handler._format_log_entry(record)
        
        # Check entry structure
        assert log_entry["level"] == "INFO"
        assert log_entry["logger"] == "test_logger"
        assert log_entry["merge_id"] == "test_merge_123"
        assert log_entry["operation_id"] == "test_op_456"
        assert log_entry["action"] == "validation"
        assert log_entry["status"] == "success"
        assert log_entry["user_id"] == "test_user"
        assert log_entry["details"] == json_data
        assert log_entry["source"] == "test_path.py:123"
        assert "timestamp" in log_entry

    def test_format_log_entry_text(self, db_handler):
        """Test formatting a log record with text message"""
        # Create a log record with text message
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_path.py",
            lineno=456,
            msg="Test error message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        log_entry = db_handler._format_log_entry(record)
        
        # Check entry structure
        assert log_entry["level"] == "ERROR"
        assert log_entry["logger"] == "test_logger"
        assert log_entry["merge_id"] == "unknown"
        assert log_entry["operation_id"] == "unknown"
        assert log_entry["action"] == "unknown"
        assert log_entry["status"] == "unknown"
        assert log_entry["details"] == {"message": "Test error message"}
        assert log_entry["source"] == "test_path.py:456"
        assert "timestamp" in log_entry

    @pytest.mark.asyncio
    async def test_store_log(self, db_handler, mock_storage):
        """Test storing a log entry in the database"""
        # Create a test log entry
        log_entry = {
            "timestamp": "2023-01-01T12:00:00",
            "level": "INFO",
            "logger": "test_logger",
            "merge_id": "test_merge_123",
            "operation_id": "test_op_456",
            "action": "validation",
            "status": "success",
            "user_id": "test_user",
            "details": {"valid": True},
            "source": "test_path.py:123",
            "process_id": 1234,
            "thread_id": 5678
        }
        
        # Set up the mock
        mock_storage.execute_query = AsyncMock()
        db_handler.storage = mock_storage
        db_handler.initialized = True
        
        # Store the log entry
        await db_handler._store_log(log_entry)
        
        # Check that execute_query was called with the correct parameters
        mock_storage.execute_query.assert_called_once()
        args, kwargs = mock_storage.execute_query.call_args
        assert "CREATE (l:MergeLog" in args[0]  # Check query
        assert args[1] == log_entry  # Check parameters

    @pytest.mark.asyncio
    async def test_emit_with_running_loop(self, db_handler, mock_storage):
        """Test emitting a log record with a running event loop"""
        with patch('app.services.storage.neo4j.Neo4jStorage', return_value=mock_storage):
            with patch('asyncio.get_event_loop') as mock_get_loop:
                # Mock the event loop
                mock_loop = MagicMock()
                mock_loop.is_running.return_value = True
                mock_get_loop.return_value = mock_loop
                
                # Create a log record
                record = logging.LogRecord(
                    name="test_logger",
                    level=logging.INFO,
                    pathname="test_path.py",
                    lineno=123,
                    msg=json.dumps({"merge_id": "test_merge_123"}),
                    args=(),
                    exc_info=None
                )
                
                # Emit the record
                with patch('asyncio.create_task') as mock_create_task:
                    db_handler.emit(record)
                    
                    # Check that create_task was called
                    mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_with_no_running_loop(self, db_handler, mock_storage):
        """Test emitting a log record without a running event loop"""
        with patch('app.services.storage.neo4j.Neo4jStorage', return_value=mock_storage):
            with patch('asyncio.get_event_loop') as mock_get_loop:
                # Mock the event loop
                mock_loop = MagicMock()
                mock_loop.is_running.return_value = False
                mock_get_loop.return_value = mock_loop
                
                # Create a log record
                record = logging.LogRecord(
                    name="test_logger",
                    level=logging.INFO,
                    pathname="test_path.py",
                    lineno=123,
                    msg=json.dumps({"merge_id": "test_merge_123"}),
                    args=(),
                    exc_info=None
                )
                
                # Emit the record
                db_handler.emit(record)
                
                # Check that run_until_complete was called
                mock_loop.run_until_complete.assert_called_once()

    def test_emit_with_exception(self, db_handler):
        """Test emitting a log record that raises an exception"""
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path.py",
            lineno=123,
            msg=json.dumps({"merge_id": "test_merge_123"}),
            args=(),
            exc_info=None
        )
        
        # Mock handleError
        db_handler.handleError = MagicMock()
        
        # Patch asyncio.get_event_loop to raise an exception
        with patch('asyncio.get_event_loop', side_effect=Exception("Test exception")):
            # Emit the record
            db_handler.emit(record)
            
            # Check that handleError was called
            db_handler.handleError.assert_called_once_with(record)

    @pytest.mark.asyncio
    async def test_store_log_with_exception(self, db_handler, mock_storage):
        """Test storing a log entry that raises an exception"""
        # Create a test log entry
        log_entry = {
            "timestamp": "2023-01-01T12:00:00",
            "level": "INFO",
            "logger": "test_logger",
            "merge_id": "test_merge_123",
            "operation_id": "test_op_456",
            "action": "validation",
            "status": "success",
            "user_id": "test_user",
            "details": {"valid": True},
            "source": "test_path.py:123",
            "process_id": 1234,
            "thread_id": 5678
        }
        
        # Set up the mock to raise an exception
        mock_storage.execute_query = AsyncMock(side_effect=Exception("Test exception"))
        db_handler.storage = mock_storage
        db_handler.initialized = True
        
        # Store the log entry (should not raise an exception)
        with patch('builtins.print') as mock_print:
            await db_handler._store_log(log_entry)
            
            # Check that the error was printed
            mock_print.assert_called_once()
            assert "Failed to store log in database" in mock_print.call_args[0][0] 