"""
Re-export of ConflictStorageInterface from app.storage.conflicts
This file exists to maintain backward compatibility with existing imports.
"""

from app.storage.conflicts import ConflictStorageInterface, ConflictStorage

__all__ = ["ConflictStorageInterface", "ConflictStorage"] 