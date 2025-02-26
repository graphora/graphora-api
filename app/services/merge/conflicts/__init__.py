"""Conflict detection and management package"""
from .service import MergeService
from .detectors import (
    EntityMatchingDetector,
    PropertyConflictDetector,
    RelationshipConflictDetector
)
from .creators import (
    EntityConflictCreator,
    PropertyConflictCreator,
    RelationshipConflictCreator
)
from .analyzers import (
    EntitySimilarityAnalyzer,
    PropertyConflictAnalyzer,
    RelationshipConflictAnalyzer
)

__all__ = [
    'MergeService',
    'EntityMatchingDetector',
    'PropertyConflictDetector',
    'RelationshipConflictDetector',
    'EntityConflictCreator',
    'PropertyConflictCreator',
    'RelationshipConflictCreator',
    'EntitySimilarityAnalyzer',
    'PropertyConflictAnalyzer',
    'RelationshipConflictAnalyzer'
]
