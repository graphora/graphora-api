from datetime import datetime


from app.services.quality.exceptions import QualityValidationError
from app.services.transform.flows import _classify_transform_failure
from app.services.transform.status_models import (
    TransformFailureReason,
    TransformationStage,
    ErrorSummary,
)
from app.services.transform.tasks import ExtractionError


class _FakeLLMError(Exception):
    def __init__(self, status=503, message="Service unavailable"):
        super().__init__(message)
        self.status = status


def test_classify_quality_no_graph_failure():
    exc = QualityValidationError(
        "Quality validation failed: no graphs generated",
        code="quality_validation_failed",
        details={"reason": "no_graphs_generated"},
    )

    classification = _classify_transform_failure(
        exc,
        TransformationStage.TRANSFORM,
        documents_processed=2,
    )

    assert classification.reason == TransformFailureReason.NO_GRAPH_GENERATED
    assert classification.code == "quality_validation_failed"
    assert classification.details["reason"] == "no_graphs_generated"
    assert classification.details["documents_processed"] == 2
    assert classification.is_recoverable is False


def test_classify_llm_unavailable_failure():
    underlying = _FakeLLMError()
    exc = ExtractionError("Knowledge graph extraction failed", underlying)

    classification = _classify_transform_failure(
        exc,
        TransformationStage.TRANSFORM,
    )

    assert classification.reason == TransformFailureReason.LLM_UNAVAILABLE
    assert classification.code == "llm_unavailable"
    assert classification.is_recoverable is True
    assert "underlying_exception" in classification.details


def test_error_summary_populates_failure_reason():
    classification_reason = TransformFailureReason.PARSE_FAILED
    error = ErrorSummary(
        stage=TransformationStage.PARSE,
        error_type="ParseError",
        error_message="Failed to parse",
        error_timestamp=datetime.utcnow(),
        failure_code="parse_failed",
        failure_reason=classification_reason,
    )

    assert error.failure_reason == classification_reason
