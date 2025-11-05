from datetime import datetime


from app.services.transform.status_models import (
    DetailedTransformStatus,
    ErrorSummary,
    ResourceMetrics,
    StageProgress,
    StageStatus,
    TransformStatus,
    TransformationStage,
)


def test_stage_progress_lifecycle_updates():
    progress = StageProgress(
        stage=TransformationStage.UPLOAD,
        status=StageStatus.PENDING,
    )

    assert progress.is_complete is False

    progress.start()
    assert progress.status == StageStatus.IN_PROGRESS

    progress.update_progress(items_processed=5, items_total=10, metrics={"chunks": 2})
    assert progress.percentage_complete == 50.0
    assert progress.metrics["chunks"] == 2

    progress.complete()
    assert progress.is_complete is True
    assert progress.status == StageStatus.COMPLETED
    assert progress.percentage_complete == 100.0


def test_detailed_transform_status_percentage_and_failure():
    stages = {
        TransformationStage.UPLOAD: StageProgress(
            stage=TransformationStage.UPLOAD, status=StageStatus.PENDING
        ),
        TransformationStage.PARSE: StageProgress(
            stage=TransformationStage.PARSE, status=StageStatus.PENDING
        ),
    }

    status = DetailedTransformStatus(
        transform_id="transform-123",
        overall_status=TransformStatus.RUNNING,
        current_stage=TransformationStage.UPLOAD,
        stages_progress=stages,
        start_time=datetime.utcnow(),
        resource_metrics=ResourceMetrics(),
    )

    status.start_stage(TransformationStage.UPLOAD)
    status.update_stage_progress(
        TransformationStage.UPLOAD, items_processed=3, items_total=6
    )
    assert (
        status.stages_progress[TransformationStage.UPLOAD].percentage_complete == 50.0
    )

    status.complete_stage(TransformationStage.UPLOAD)
    assert (
        status.stages_progress[TransformationStage.UPLOAD].status
        == StageStatus.COMPLETED
    )
    assert status.percentage_complete == 50.0

    error = ErrorSummary(
        stage=TransformationStage.PARSE,
        error_type="ParseError",
        error_message="Failed to parse",
        error_timestamp=datetime.utcnow(),
    )

    status.fail_stage(TransformationStage.PARSE, error)

    assert status.overall_status == TransformStatus.FAILED
    assert status.current_stage == TransformationStage.FAILED
    assert status.error_summary.error_message == "Failed to parse"
    assert (
        status.stages_progress[TransformationStage.PARSE].status == StageStatus.FAILED
    )
    assert (
        status.stages_progress[TransformationStage.PARSE].error_details["error_message"]
        == "Failed to parse"
    )
    assert status.duration_ms >= 0.0
    assert status.failure_reason is None
