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
from app.services.merge.models import MergeStage, StageStatus

# Import resolution learning service
from app.services.merge.resolution_learning import ResolutionLearningService, ResolutionLearningConfig

__all__ = [
    "MergeService",
    "create_resolution_pipeline_deployment",
    "run_resolution_pipeline",
    "get_flow_run_status",
    "cancel_flow_run",
    "ProgressTracker",
    "MergeStage",
    "StageStatus",
    "ResolutionLearningService",
    "ResolutionLearningConfig"
] 