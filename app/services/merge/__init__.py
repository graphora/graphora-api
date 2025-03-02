"""Merge service module for handling merge operations and conflict resolution."""

# Import the main service class
from app.services.merge.service import MergeService

# Import flow management modules
from app.services.merge.flow_manager import (
    create_resolution_pipeline_deployment,
    run_resolution_pipeline,
    get_flow_run_status,
    cancel_flow_run
)

# Import progress tracking
from app.services.merge.progress import ProgressTracker

# Import models
from app.services.merge.models import (
    MergeStage, MergeStatus, StageStatus, MergeProgress, 
    ValidationResult, ValidationIssue, ValidationSeverity, ValidationIssueType,
    EntityMappingResult, EntityMatch, MatchStrategy,
    RollbackType, RollbackOptions, RollbackResponse, SnapshotData
)

# Import resolution learning service
from app.services.merge.resolution_learning import ResolutionLearningService, ResolutionLearningConfig

# Import conflict detection service
from app.services.merge.conflict import ConflictDetectionService

# Import merge validation service
from app.services.merge.validation import MergeValidationService

# Import merge execution service
from app.services.merge.execution_service import MergeExecutionService

# Import merge logger
from app.services.merge.merge_logger import MergeLogger

__all__ = [
    "MergeService",
    "ConflictDetectionService",
    "MergeValidationService",
    "MergeExecutionService",
    "create_resolution_pipeline_deployment",
    "run_resolution_pipeline",
    "get_flow_run_status",
    "cancel_flow_run",
    "ProgressTracker",
    "MergeStage",
    "MergeStatus",
    "StageStatus",
    "MergeProgress",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationIssueType",
    "EntityMappingResult",
    "EntityMatch",
    "MatchStrategy",
    "RollbackType",
    "RollbackOptions",
    "RollbackResponse",
    "SnapshotData",
    "MergeLogger",
    "ResolutionLearningService",
    "ResolutionLearningConfig"
] 