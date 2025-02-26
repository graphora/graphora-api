"""Custom exceptions for storage operations"""

class StorageError(Exception):
    """Base class for storage exceptions"""
    pass

class StorageConnectionError(StorageError):
    """Raised when connection to storage fails"""
    pass

class StorageAuthError(StorageError):
    """Raised when authentication fails"""
    pass

class StorageQueryError(StorageError):
    """Raised when query execution fails"""
    pass
