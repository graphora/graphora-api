"""Quality service for managing quality validation results."""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime, timezone
import hashlib
import json
import csv
import io
from collections import Counter

from graphora_server.utils.logger import logger

from .models import (
    QualityResults,
    QualityViolation,
    QualitySeverity,
    QualityRuleType,
)

if (
    TYPE_CHECKING
):  # pragma: no cover - import only for typing to avoid heavy deps at runtime
    from graphora_server.services.storage.neo4j import Neo4jStorage


class QualityService:
    """Service for managing quality validation results and user interactions."""

    def __init__(self, neo4j_storage: "Neo4jStorage"):
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

            await self._store_quality_violations(
                transform_id=transform_id,
                user_id=user_id,
                violations=quality_results.violations,
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
        violation_type: Optional[QualityRuleType | str] = None,
        severity: Optional[QualitySeverity | str] = None,
        entity_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        quality_results: Optional[QualityResults] = None,
    ) -> List[QualityViolation]:
        """Get filtered list of quality violations."""
        try:
            if limit <= 0:
                return []

            if offset < 0:
                offset = 0

            if quality_results is None:
                quality_results = await self.get_quality_results(transform_id, user_id)

            if not quality_results:
                return []

            violations = quality_results.violations

            if violation_type:
                violation_type_value = (
                    violation_type.value
                    if isinstance(violation_type, QualityRuleType)
                    else str(violation_type)
                )
                violations = [
                    v
                    for v in violations
                    if getattr(v.rule_type, "value", v.rule_type)
                    == violation_type_value
                ]

            if severity:
                severity_value = (
                    severity.value
                    if isinstance(severity, QualitySeverity)
                    else str(severity)
                )
                violations = [
                    v
                    for v in violations
                    if getattr(v.severity, "value", v.severity) == severity_value
                ]

            if entity_type:
                violations = [v for v in violations if v.entity_type == entity_type]

            return violations[offset : offset + limit]
        except Exception as e:
            logger.error(f"Failed to get violations for transform {transform_id}: {e}")
            return []

    async def _store_quality_violations(
        self,
        *,
        transform_id: str,
        user_id: str,
        violations: List[QualityViolation],
    ) -> None:
        """Persist individual quality violations for analytics and UI triage."""

        delete_query = """
        MATCH (qr:QualityResults {transform_id: $transform_id, user_id: $user_id})-[:HAS_VIOLATION]->(v:QualityViolation)
        DETACH DELETE v
        """

        params = {"transform_id": transform_id, "user_id": user_id}

        try:
            await self.neo4j.execute_query(delete_query, params)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Failed to clear existing quality violations for transform %s: %s",
                transform_id,
                exc,
            )
            return

        if not violations:
            logger.debug(
                "No quality violations to persist for transform %s", transform_id
            )
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        violations_payload: List[Dict[str, Any]] = []

        for index, violation in enumerate(violations):
            violation_dict = violation.model_dump(mode="json")
            identifier_source = json.dumps(
                {
                    "rule_id": violation_dict.get("rule_id"),
                    "entity_id": violation_dict.get("entity_id"),
                    "property_name": violation_dict.get("property_name"),
                    "relationship_type": violation_dict.get("relationship_type"),
                    "index": index,
                },
                sort_keys=True,
            )
            violation_id = hashlib.sha1(
                f"{transform_id}:{identifier_source}".encode("utf-8")
            ).hexdigest()

            violations_payload.append(
                {
                    **violation_dict,
                    "violation_id": violation_id,
                    "violation_index": index,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                }
            )

        create_query = """
        MATCH (qr:QualityResults {transform_id: $transform_id, user_id: $user_id})
        WITH qr, $violations AS violations
        UNWIND violations AS violation
        CREATE (qr)-[:HAS_VIOLATION]->(v:QualityViolation {violation_id: violation.violation_id})
        SET v.rule_id = violation.rule_id,
            v.rule_type = violation.rule_type,
            v.severity = violation.severity,
            v.entity_type = violation.entity_type,
            v.entity_id = violation.entity_id,
            v.property_name = violation.property_name,
            v.relationship_type = violation.relationship_type,
            v.message = violation.message,
            v.expected = violation.expected,
            v.actual = violation.actual,
            v.confidence = violation.confidence,
            v.suggestion = violation.suggestion,
            v.context = violation.context,
            v.violation_index = violation.violation_index,
            v.created_at = violation.created_at,
            v.updated_at = violation.updated_at
        """

        try:
            await self.neo4j.execute_query(
                create_query,
                {
                    "transform_id": transform_id,
                    "user_id": user_id,
                    "violations": violations_payload,
                },
            )
            logger.debug(
                "Stored %s quality violations for transform %s",
                len(violations_payload),
                transform_id,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Failed to persist quality violations for transform %s: %s",
                transform_id,
                exc,
            )

    async def get_violation_analytics(
        self, transform_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return aggregated analytics for a transform's quality results."""

        results = await self.get_quality_results(transform_id, user_id)
        if not results:
            return None

        def _normalize_enum_dict(data: Dict[Any, int]) -> Dict[str, int]:
            normalized: Dict[str, int] = {}
            for key, value in (data or {}).items():
                if hasattr(key, "value"):
                    normalized[str(key.value)] = value
                else:
                    normalized[str(key)] = value
            return normalized

        metrics_dict = results.metrics.model_dump()
        by_severity = _normalize_enum_dict(results.violations_by_severity)
        by_type = _normalize_enum_dict(results.violations_by_type)
        by_entity = {
            str(entity): count
            for entity, count in (results.violations_by_entity_type or {}).items()
        }

        rule_counter: Counter[str] = Counter()
        rule_metadata: Dict[str, Dict[str, Any]] = {}
        entity_offenders: Counter[str] = Counter()

        for violation in results.violations:
            rule_counter[violation.rule_id] += 1
            entity_offenders[violation.entity_type or "unknown"] += 1
            if violation.rule_id not in rule_metadata:
                rule_metadata[violation.rule_id] = {
                    "severity": (
                        violation.severity.value
                        if isinstance(violation.severity, QualitySeverity)
                        else str(violation.severity)
                    ),
                    "rule_type": (
                        violation.rule_type.value
                        if isinstance(violation.rule_type, QualityRuleType)
                        else str(violation.rule_type)
                    ),
                }

        top_rules = [
            {
                "rule_id": rule_id,
                "count": count,
                **rule_metadata.get(rule_id, {}),
            }
            for rule_id, count in rule_counter.most_common(10)
        ]

        top_entities = [
            {"entity_type": entity_type, "count": count}
            for entity_type, count in entity_offenders.most_common(10)
        ]

        analytics: Dict[str, Any] = {
            "transform_id": transform_id,
            "overall_score": results.overall_score,
            "grade": results.grade,
            "quality_gate_status": results.quality_gate_status,
            "requires_review": results.requires_review,
            "metrics": metrics_dict,
            "violations": {
                "total": len(results.violations),
                "by_severity": by_severity,
                "by_type": by_type,
                "by_entity_type": by_entity,
                "top_rules": top_rules,
                "top_entities": top_entities,
            },
            "property_fill_rates": {
                str(entity): rate
                for entity, rate in results.metrics.property_fill_rates_by_entity.items()
            },
            "entity_type_coverage": {
                str(entity): count
                for entity, count in results.metrics.entity_type_coverage.items()
            },
        }

        return analytics

    async def export_violations_csv(
        self, transform_id: str, user_id: str
    ) -> Optional[str]:
        """Export violations for a transform as CSV text."""

        results = await self.get_quality_results(transform_id, user_id)
        if not results or not results.violations:
            return None

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "rule_id",
                "rule_type",
                "severity",
                "entity_type",
                "entity_id",
                "property_name",
                "relationship_type",
                "message",
                "expected",
                "actual",
                "confidence",
                "suggestion",
            ]
        )

        for violation in results.violations:
            writer.writerow(
                [
                    violation.rule_id,
                    (
                        violation.rule_type.value
                        if isinstance(violation.rule_type, QualityRuleType)
                        else str(violation.rule_type)
                    ),
                    (
                        violation.severity.value
                        if isinstance(violation.severity, QualitySeverity)
                        else str(violation.severity)
                    ),
                    violation.entity_type or "",
                    violation.entity_id or "",
                    violation.property_name or "",
                    violation.relationship_type or "",
                    violation.message,
                    violation.expected,
                    violation.actual,
                    f"{violation.confidence:.2f}",
                    violation.suggestion or "",
                ]
            )

        return buffer.getvalue()

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
