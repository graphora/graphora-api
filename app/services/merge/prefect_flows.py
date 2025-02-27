"""Prefect flows for merge conflict resolution"""
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.schemas.conflicts import Conflict, ConflictSeverity, ConflictType
from app.services.merge.models import MergeStage, StageStatus
from app.config import settings

@task(name="load_conflicts", retries=2, retry_delay_seconds=10)
async def load_conflicts(
    merge_id: str,
    conflict_type: Optional[str] = None,
    severity: Optional[str] = None
) -> List[Conflict]:
    """Load conflicts from storage for resolution"""
    # Import here to avoid circular imports
    from app.services.merge.service import MergeService
    
    merge_service = MergeService()
    conflicts, total = await merge_service.get_conflicts(
        merge_id=merge_id,
        conflict_type=conflict_type, 
        severity=severity,
        resolved=False
    )
    
    return conflicts

@task(name="resolve_minor_conflicts", retries=3, retry_delay_seconds=5)
async def resolve_minor_conflicts(
    merge_id: str,
    conflicts: List[Conflict]
) -> Dict[str, Any]:
    """Automatically resolve minor conflicts"""
    # Import here to avoid circular imports
    from app.services.merge.service import MergeService
    
    minor_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.MINOR]
    
    merge_service = MergeService()
    results = {}
    
    for conflict in minor_conflicts:
        # Find highest confidence resolution option
        if not conflict.resolution_options:
            results[conflict.id] = {"status": "skipped", "reason": "no_options"}
            continue
            
        best_option = max(
            conflict.resolution_options,
            key=lambda opt: opt.confidence
        )
        
        try:
            # Apply resolution
            resolved = await merge_service.apply_resolution(
                merge_id=merge_id,
                conflict_id=conflict.id,
                resolution_id=best_option.id
            )
            
            results[conflict.id] = {
                "status": "resolved" if resolved else "failed",
                "resolution_id": best_option.id,
                "confidence": best_option.confidence
            }
        except Exception as e:
            results[conflict.id] = {
                "status": "error",
                "error": str(e)
            }
    
    return {
        "total": len(minor_conflicts),
        "resolved": sum(1 for r in results.values() if r.get("status") == "resolved"),
        "failed": sum(1 for r in results.values() if r.get("status") in ["failed", "error"]),
        "skipped": sum(1 for r in results.values() if r.get("status") == "skipped"),
        "results": results
    }

@task(name="resolve_major_conflicts", retries=2, retry_delay_seconds=5)
async def resolve_major_conflicts(
    merge_id: str,
    conflicts: List[Conflict]
) -> Dict[str, Any]:
    """Apply LLM-assisted resolution to major conflicts"""
    # Import here to avoid circular imports
    from app.services.merge.service import MergeService
    
    major_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.MAJOR]
    
    merge_service = MergeService()
    results = {}
    
    # First analyze conflicts with LLM if they don't have resolution options
    conflicts_to_analyze = [c for c in major_conflicts if not c.resolution_options]
    if conflicts_to_analyze:
        await merge_service.analyze_conflicts_with_llm(
            merge_id=merge_id, 
            conflicts=[c.id for c in conflicts_to_analyze]
        )
        
        # Reload conflicts with updated options
        if conflicts_to_analyze:
            for i, conflict in enumerate(major_conflicts):
                if conflict.id in [c.id for c in conflicts_to_analyze]:
                    updated_conflict = await merge_service.get_conflict(merge_id, conflict.id)
                    if updated_conflict:
                        major_conflicts[i] = updated_conflict
    
    # Apply resolutions
    for conflict in major_conflicts:
        if not conflict.resolution_options:
            results[conflict.id] = {"status": "skipped", "reason": "no_options"}
            continue
            
        # Get highest confidence option above threshold
        confidence_threshold = 0.7
        high_confidence_options = [
            opt for opt in conflict.resolution_options
            if opt.confidence >= confidence_threshold
        ]
        
        if not high_confidence_options:
            results[conflict.id] = {
                "status": "skipped", 
                "reason": "low_confidence",
                "highest_confidence": max(
                    (opt.confidence for opt in conflict.resolution_options),
                    default=0
                )
            }
            continue
            
        best_option = max(
            high_confidence_options,
            key=lambda opt: opt.confidence
        )
        
        try:
            # Apply resolution
            resolved = await merge_service.apply_resolution(
                merge_id=merge_id,
                conflict_id=conflict.id,
                resolution_id=best_option.id
            )
            
            results[conflict.id] = {
                "status": "resolved" if resolved else "failed",
                "resolution_id": best_option.id,
                "confidence": best_option.confidence
            }
        except Exception as e:
            results[conflict.id] = {
                "status": "error",
                "error": str(e)
            }
    
    return {
        "total": len(major_conflicts),
        "resolved": sum(1 for r in results.values() if r.get("status") == "resolved"),
        "failed": sum(1 for r in results.values() if r.get("status") in ["failed", "error"]),
        "skipped": sum(1 for r in results.values() if r.get("status") == "skipped"),
        "results": results
    }

@task(name="flag_critical_conflicts")
async def flag_critical_conflicts(
    merge_id: str,
    conflicts: List[Conflict]
) -> Dict[str, Any]:
    """Flag critical conflicts for human review"""
    # Import here to avoid circular imports
    from app.services.merge.service import MergeService
    
    critical_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.CRITICAL]
    
    merge_service = MergeService()
    
    # Mark these conflicts as requiring human review
    for conflict in critical_conflicts:
        await merge_service.mark_conflict_for_review(
            merge_id=merge_id,
            conflict_id=conflict.id
        )
    
    return {
        "total_critical": len(critical_conflicts),
        "flagged_for_review": len(critical_conflicts),
        "conflict_ids": [c.id for c in critical_conflicts]
    }

@task(name="check_resolution_status")
async def check_resolution_status(
    merge_id: str
) -> Dict[str, Any]:
    """Check overall resolution status"""
    # Import here to avoid circular imports
    from app.services.merge.service import MergeService
    
    merge_service = MergeService()
    
    # Get all conflicts
    all_conflicts, total = await merge_service.get_conflicts(
        merge_id=merge_id
    )
    
    # Count resolved vs unresolved
    resolved = [c for c in all_conflicts if c.resolved]
    unresolved = [c for c in all_conflicts if not c.resolved]
    
    # Check if human review is needed
    human_review_needed = any(
        c.severity == ConflictSeverity.CRITICAL and not c.resolved
        for c in all_conflicts
    )
    
    # Check if all auto-resolvable conflicts are resolved
    auto_resolvable = [
        c for c in all_conflicts 
        if c.severity != ConflictSeverity.CRITICAL
    ]
    auto_resolved = [
        c for c in auto_resolvable
        if c.resolved
    ]
    
    auto_resolution_complete = len(auto_resolvable) == len(auto_resolved)
    
    return {
        "total_conflicts": total,
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "resolution_percentage": (len(resolved) / total) * 100 if total > 0 else 0,
        "human_review_needed": human_review_needed,
        "auto_resolution_complete": auto_resolution_complete,
        "critical_unresolved": [
            c.id for c in unresolved 
            if c.severity == ConflictSeverity.CRITICAL
        ]
    }

@flow(
    name="resolution-pipeline",
    description="Conflict resolution workflow",
    task_runner=ConcurrentTaskRunner(),
    retries=1,
    version="1.0.0"
)
async def resolution_pipeline_flow(
    merge_id: str
) -> Dict[str, Any]:
    """Main resolution workflow"""
    # Import here to avoid circular imports
    from app.services.merge.service import MergeService
    
    merge_service = MergeService()
    
    # Start resolution stage
    await merge_service.progress_tracker.start_merge_stage(
        merge_id, 
        MergeStage.RESOLUTION
    )
    
    try:
        # Load all conflicts
        conflicts = await load_conflicts(merge_id)
        
        if not conflicts:
            # No conflicts to resolve
            await merge_service.progress_tracker.complete_merge_stage(
                merge_id, 
                MergeStage.RESOLUTION
            )
            
            return {
                "status": "completed",
                "message": "No conflicts to resolve",
                "merge_id": merge_id
            }
        
        # Update progress
        await merge_service.progress_tracker.update_merge_progress(
            merge_id,
            MergeStage.RESOLUTION,
            1, 5,
            {"task": "loaded_conflicts"}
        )
        
        # Process different severity levels
        minor_results = await resolve_minor_conflicts(merge_id, conflicts)
        
        # Update progress
        await merge_service.progress_tracker.update_merge_progress(
            merge_id,
            MergeStage.RESOLUTION,
            2, 5,
            {"task": "resolved_minor_conflicts"}
        )
        
        major_results = await resolve_major_conflicts(merge_id, conflicts)
        
        # Update progress
        await merge_service.progress_tracker.update_merge_progress(
            merge_id,
            MergeStage.RESOLUTION,
            3, 5,
            {"task": "resolved_major_conflicts"}
        )
        
        critical_results = await flag_critical_conflicts(merge_id, conflicts)
        
        # Update progress
        await merge_service.progress_tracker.update_merge_progress(
            merge_id,
            MergeStage.RESOLUTION,
            4, 5,
            {"task": "flagged_critical_conflicts"}
        )
        
        # Check final status
        status = await check_resolution_status(merge_id)
        
        # Update progress
        await merge_service.progress_tracker.update_merge_progress(
            merge_id,
            MergeStage.RESOLUTION,
            5, 5,
            {"task": "checked_resolution_status"}
        )
        
        # Complete or pause for human review
        if status["human_review_needed"]:
            # Mark as waiting for human review
            await merge_service.progress_tracker.pause_merge_stage(
                merge_id, 
                MergeStage.RESOLUTION,
                reason="waiting_for_human_review",
                details={"critical_conflicts": status["critical_unresolved"]}
            )
            
            return {
                "status": "paused",
                "message": "Waiting for human review of critical conflicts",
                "merge_id": merge_id,
                "requires_review": status["critical_unresolved"],
                "minor_results": minor_results,
                "major_results": major_results,
                "critical_results": critical_results,
                "overall_status": status
            }
        else:
            # Complete resolution stage
            await merge_service.progress_tracker.complete_merge_stage(
                merge_id, 
                MergeStage.RESOLUTION
            )
            
            return {
                "status": "completed",
                "message": "All conflicts resolved automatically",
                "merge_id": merge_id,
                "minor_results": minor_results,
                "major_results": major_results,
                "critical_results": critical_results,
                "overall_status": status
            }
    except Exception as e:
        # Report failure
        await merge_service.progress_tracker.fail_merge_stage(
            merge_id, 
            MergeStage.RESOLUTION,
            str(e)
        )
        
        raise 