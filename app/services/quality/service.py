"""Quality service for managing quality validation results."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from app.services.storage.neo4j import Neo4jStorage
from app.utils.logger import logger

from .models import QualityResults, QualityViolation, QualitySeverity


class QualityService:
    """Service for managing quality validation results and user interactions."""

    def __init__(self, neo4j_storage: Neo4jStorage):
        self.neo4j = neo4j_storage

    async def store_quality_results(
        self, transform_id: str, quality_results: QualityResults, user_id: str
    ) -> None:
        """Store quality validation results for later retrieval."""
        try:
            # Serialize quality results to JSON
            results_json = quality_results.model_dump_json()

            # Store in Neo4j or your preferred storage
            query = """
            MERGE (qr:QualityResults {transform_id: $transform_id, user_id: $user_id})
            SET qr.results_json = $results_json,
                qr.overall_score = $overall_score,
                qr.grade = $grade,
                qr.requires_review = $requires_review,
                qr.total_violations = $total_violations,
                qr.created_at = $created_at,
                qr.updated_at = $updated_at
            RETURN qr
            """

            await self.neo4j.execute_query(
                query,
                {
                    "transform_id": transform_id,
                    "user_id": user_id,
                    "results_json": results_json,
                    "overall_score": quality_results.overall_score,
                    "grade": quality_results.grade,
                    "requires_review": quality_results.requires_review,
                    "total_violations": len(quality_results.violations),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info(
                f"Successfully stored quality results for transform {transform_id}: "
                f"Score={quality_results.overall_score:.1f}, Grade={quality_results.grade}, "
                f"Violations={len(quality_results.violations)}"
            )

        except Exception as e:
            logger.error(
                f"Failed to store quality results for transform {transform_id}: {e}"
            )
            raise

    async def get_quality_results(
        self, transform_id: str, user_id: str
    ) -> Optional[QualityResults]:
        """Retrieve quality validation results."""
        try:
            query = """
            MATCH (qr:QualityResults {transform_id: $transform_id, user_id: $user_id})
            RETURN qr.results_json as results_json
            """

            result = await self.neo4j.execute_query(
                query, {"transform_id": transform_id, "user_id": user_id}
            )

            if not result or not result[0]:
                logger.warning(
                    f"No quality results found in Neo4j for transform {transform_id}, user {user_id}"
                )
                return None

            results_json = result[0].get("results_json")
            if not results_json:
                return None

            # Deserialize from JSON
            quality_results = QualityResults.model_validate_json(results_json)
            return quality_results

        except Exception as e:
            logger.error(
                f"Failed to retrieve quality results for transform {transform_id}: {e}"
            )
            return None

    async def approve_quality_results(
        self, transform_id: str, user_id: str, approval_comment: Optional[str] = None
    ) -> None:
        """Mark quality results as approved by user."""
        try:
            query = """
            MATCH (qr:QualityResults {transform_id: $transform_id, user_id: $user_id})
            SET qr.approved = true,
                qr.approved_at = $approved_at,
                qr.approval_comment = $approval_comment,
                qr.updated_at = $updated_at
            RETURN qr
            """

            await self.neo4j.execute_query(
                query,
                {
                    "transform_id": transform_id,
                    "user_id": user_id,
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                    "approval_comment": approval_comment,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info(
                f"Quality results approved for transform {transform_id} by user {user_id}"
            )

        except Exception as e:
            logger.error(
                f"Failed to approve quality results for transform {transform_id}: {e}"
            )
            raise

    async def reject_quality_results(
        self, transform_id: str, rejection_reason: str, user_id: str
    ) -> None:
        """Mark quality results as rejected by user."""
        try:
            query = """
            MATCH (qr:QualityResults {transform_id: $transform_id, user_id: $user_id})
            SET qr.rejected = true,
                qr.rejected_at = $rejected_at,
                qr.rejection_reason = $rejection_reason,
                qr.updated_at = $updated_at
            RETURN qr
            """

            await self.neo4j.execute_query(
                query,
                {
                    "transform_id": transform_id,
                    "user_id": user_id,
                    "rejected_at": datetime.now(timezone.utc).isoformat(),
                    "rejection_reason": rejection_reason,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info(
                f"Quality results rejected for transform {transform_id} by user {user_id}: {rejection_reason}"
            )

        except Exception as e:
            logger.error(
                f"Failed to reject quality results for transform {transform_id}: {e}"
            )
            raise

    async def get_violations(
        self,
        transform_id: str,
        user_id: str,
        violation_type: Optional[str] = None,
        severity: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[QualityViolation]:
        """Get filtered list of quality violations."""
        try:
            quality_results = await self.get_quality_results(transform_id, user_id)
            if not quality_results:
                return []

            violations = quality_results.violations

            # Apply filters
            if violation_type:
                violations = [v for v in violations if v.rule_type == violation_type]

            if severity:
                violations = [v for v in violations if v.severity == severity]

            if entity_type:
                violations = [v for v in violations if v.entity_type == entity_type]

            # Apply pagination
            return violations[offset : offset + limit]

        except Exception as e:
            logger.error(f"Failed to get violations for transform {transform_id}: {e}")
            return []

    async def get_quality_summary(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get summary of recent quality results for a user."""
        try:
            query = """
            MATCH (qr:QualityResults {user_id: $user_id})
            RETURN qr.transform_id as transform_id,
                   qr.overall_score as overall_score,
                   qr.grade as grade,
                   qr.requires_review as requires_review,
                   qr.total_violations as total_violations,
                   qr.created_at as created_at,
                   qr.approved as approved,
                   qr.rejected as rejected
            ORDER BY qr.created_at DESC
            LIMIT $limit
            """

            results = await self.neo4j.execute_query(
                query, {"user_id": user_id, "limit": limit}
            )

            summaries = []
            for result in results:
                summary = {
                    "transform_id": result.get("transform_id"),
                    "overall_score": result.get("overall_score"),
                    "grade": result.get("grade"),
                    "requires_review": result.get("requires_review"),
                    "total_violations": result.get("total_violations"),
                    "created_at": result.get("created_at"),
                    "status": (
                        "approved"
                        if result.get("approved")
                        else ("rejected" if result.get("rejected") else "pending")
                    ),
                }
                summaries.append(summary)

            return summaries

        except Exception as e:
            logger.error(f"Failed to get quality summary for user {user_id}: {e}")
            return []

    async def delete_quality_results(self, transform_id: str, user_id: str) -> bool:
        """Delete quality results for a transform."""
        try:
            query = """
            MATCH (qr:QualityResults {transform_id: $transform_id, user_id: $user_id})
            DELETE qr
            RETURN count(qr) as deleted_count
            """

            result = await self.neo4j.execute_query(
                query, {"transform_id": transform_id, "user_id": user_id}
            )

            deleted_count = result[0].get("deleted_count", 0) if result else 0

            if deleted_count > 0:
                logger.info(f"Deleted quality results for transform {transform_id}")
                return True
            else:
                logger.warning(
                    f"No quality results found to delete for transform {transform_id}"
                )
                return False

        except Exception as e:
            logger.error(
                f"Failed to delete quality results for transform {transform_id}: {e}"
            )
            return False

    def calculate_auto_approval_eligibility(
        self,
        quality_results: QualityResults,
        auto_approval_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check if quality results are eligible for auto-approval."""
        if not auto_approval_config:
            auto_approval_config = {
                "auto_approve_threshold": 95.0,
                "max_warnings": 5,
                "no_errors_required": True,
            }

        eligible = True
        reasons = []

        gate_status = getattr(quality_results, "quality_gate_status", "pass")
        if gate_status == "fail":
            eligible = False
            reasons.append("Quality gate status is fail")
        elif gate_status == "warn":
            eligible = False
            reasons.append("Quality gate reported warnings")

        # Check score threshold
        threshold = auto_approval_config.get("auto_approve_threshold", 95.0)
        if quality_results.overall_score < threshold:
            eligible = False
            reasons.append(
                f"Score {quality_results.overall_score:.1f} below threshold {threshold}"
            )

        # Check error count
        if auto_approval_config.get("no_errors_required", True):
            error_count = quality_results.violations_by_severity.get(
                QualitySeverity.ERROR, 0
            )
            if error_count > 0:
                eligible = False
                reasons.append(f"{error_count} error(s) found")

        # Check warning count
        max_warnings = auto_approval_config.get("max_warnings", 5)
        warning_count = quality_results.violations_by_severity.get(
            QualitySeverity.WARNING, 0
        )
        if warning_count > max_warnings:
            eligible = False
            reasons.append(f"{warning_count} warnings exceed limit of {max_warnings}")

        return {
            "eligible": eligible,
            "reasons": reasons,
            "config_used": auto_approval_config,
        }
