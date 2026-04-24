from typing import Any, Dict, Optional


class QualityValidationError(Exception):
    """Base exception for quality validation failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "quality_validation_failed",
        details: Optional[Dict[str, Any]] = None,
        retry_allowed: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.retry_allowed = retry_allowed


class QualityThresholdNotMetError(QualityValidationError):
    """Raised when the overall quality score falls below the minimum threshold."""

    def __init__(
        self,
        message: str,
        *,
        score: float,
        threshold: float,
        violations: Optional[Any] = None,
        quality_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        details: Dict[str, Any] = {
            "quality_score": score,
            "quality_threshold": threshold,
        }
        if violations is not None:
            details["violations"] = violations
        if quality_results is not None:
            details["quality_results"] = quality_results
        super().__init__(
            message,
            code="quality_threshold_not_met",
            details=details,
            retry_allowed=False,
        )


class QualityViolationError(QualityValidationError):
    """Raised when blocking quality violations are detected."""

    def __init__(
        self,
        message: str,
        *,
        violations: Optional[Any] = None,
        quality_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        details: Dict[str, Any] = {}
        if violations is not None:
            details["violations"] = violations
        if quality_results is not None:
            details["quality_results"] = quality_results
        super().__init__(
            message,
            code="quality_validation_failed",
            details=details,
            retry_allowed=False,
        )
