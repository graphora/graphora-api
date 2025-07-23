"""Prefect tasks for quality validation."""

from prefect import task
from typing import Dict, Any, Optional
import logging

from app.services.transform.models import DocumentKnowledgeGraph
from app.services.quality.validator import QualityValidator
from app.services.quality.models import QualityResults
from app.services.quality.service import QualityService
from app.services.storage.neo4j import Neo4jService

logger = logging.getLogger(__name__)


@task(
    name="quality_validation_task",
    description="Validate extracted knowledge graph against quality rules",
    retries=2,
    retry_delay_seconds=30
)
async def quality_validation_task(
    knowledge_graph: DocumentKnowledgeGraph,
    ontology_with_rules: Dict[str, Any],
    transform_id: str,
    user_id: str
) -> QualityResults:
    """
    Prefect task to validate extracted knowledge graph against quality rules.
    
    Args:
        knowledge_graph: The extracted knowledge graph to validate
        ontology_with_rules: Ontology schema including quality rules
        transform_id: ID of the transform being validated
        user_id: ID of the user who initiated the transform
        
    Returns:
        QualityResults: Complete quality validation results
    """
    try:
        logger.info(f"Starting quality validation task for transform {transform_id}")
        
        # Initialize quality validator
        validator = QualityValidator(ontology_with_rules)
        
        # Perform validation
        quality_results = await validator.validate_extraction(knowledge_graph, transform_id)
        
        # Store results for user review
        neo4j_service = Neo4jService()
        quality_service = QualityService(neo4j_service)
        await quality_service.store_quality_results(transform_id, quality_results, user_id)
        
        logger.info(
            f"Quality validation completed for transform {transform_id}: "
            f"Score={quality_results.overall_score:.1f}, "
            f"Grade={quality_results.grade}, "
            f"Violations={len(quality_results.violations)}"
        )
        
        return quality_results
        
    except Exception as e:
        logger.error(f"Quality validation task failed for transform {transform_id}: {e}")
        raise


@task(
    name="auto_approval_check_task", 
    description="Check if quality results are eligible for auto-approval",
    retries=1
)
async def auto_approval_check_task(
    quality_results: QualityResults,
    transform_id: str,
    user_id: str,
    auto_approval_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Check if quality results meet auto-approval criteria.
    
    Args:
        quality_results: Quality validation results
        transform_id: ID of the transform
        user_id: ID of the user  
        auto_approval_config: Configuration for auto-approval thresholds
        
    Returns:
        Dict containing auto-approval decision and reasoning
    """
    try:
        logger.info(f"Checking auto-approval eligibility for transform {transform_id}")
        
        # Initialize quality service
        neo4j_service = Neo4jService()
        quality_service = QualityService(neo4j_service)
        
        # Check auto-approval eligibility
        eligibility = quality_service.calculate_auto_approval_eligibility(
            quality_results, auto_approval_config
        )
        
        result = {
            'transform_id': transform_id,
            'auto_approve': eligibility['eligible'],
            'reasons': eligibility['reasons'],
            'quality_score': quality_results.overall_score,
            'requires_manual_review': quality_results.requires_review
        }
        
        if eligibility['eligible']:
            logger.info(f"Transform {transform_id} eligible for auto-approval")
            # Auto-approve the results
            await quality_service.approve_quality_results(
                transform_id, user_id, "Auto-approved based on quality score"
            )
            result['auto_approved'] = True
        else:
            logger.info(f"Transform {transform_id} requires manual review: {', '.join(eligibility['reasons'])}")
            result['auto_approved'] = False
        
        return result
        
    except Exception as e:
        logger.error(f"Auto-approval check failed for transform {transform_id}: {e}")
        return {
            'transform_id': transform_id,
            'auto_approve': False,
            'auto_approved': False,
            'error': str(e),
            'requires_manual_review': True
        }


@task(
    name="quality_metrics_calculation_task",
    description="Calculate detailed quality metrics for reporting"
)
async def quality_metrics_calculation_task(
    quality_results: QualityResults,
    transform_id: str
) -> Dict[str, Any]:
    """
    Calculate additional quality metrics for detailed reporting.
    
    Args:
        quality_results: Quality validation results
        transform_id: ID of the transform
        
    Returns:
        Dict containing detailed quality metrics
    """
    try:
        logger.info(f"Calculating detailed quality metrics for transform {transform_id}")
        
        violations = quality_results.violations
        metrics = quality_results.metrics
        
        # Calculate additional metrics
        detailed_metrics = {
            'transform_id': transform_id,
            'basic_metrics': {
                'total_entities': metrics.total_entities,
                'total_relationships': metrics.total_relationships,
                'overall_score': quality_results.overall_score,
                'grade': quality_results.grade
            },
            'violation_analysis': {
                'total_violations': len(violations),
                'by_severity': dict(quality_results.violations_by_severity),
                'by_type': dict(quality_results.violations_by_type),
                'by_entity_type': dict(quality_results.violations_by_entity_type)
            },
            'confidence_analysis': {
                'avg_entity_confidence': metrics.avg_entity_confidence,
                'confidence_by_type': metrics.confidence_scores_by_type,
                'low_confidence_entities': [
                    entity_type for entity_type, confidence in metrics.confidence_scores_by_type.items()
                    if confidence < 0.7
                ]
            },
            'completeness_analysis': {
                'property_completeness_rate': metrics.property_completeness_rate,
                'entity_type_coverage': metrics.entity_type_coverage
            },
            'recommendations': _generate_quality_recommendations(quality_results)
        }
        
        return detailed_metrics
        
    except Exception as e:
        logger.error(f"Quality metrics calculation failed for transform {transform_id}: {e}")
        return {'transform_id': transform_id, 'error': str(e)}


def _generate_quality_recommendations(quality_results: QualityResults) -> List[str]:
    """Generate recommendations based on quality results."""
    recommendations = []
    
    # Score-based recommendations
    if quality_results.overall_score < 70:
        recommendations.append("Consider reviewing the ontology schema for better extraction guidance")
    
    # Violation-based recommendations
    error_count = quality_results.violations_by_severity.get('error', 0)
    if error_count > 0:
        recommendations.append(f"Address {error_count} critical error(s) before proceeding")
    
    warning_count = quality_results.violations_by_severity.get('warning', 0)
    if warning_count > 10:
        recommendations.append("High number of warnings - consider refining quality rules")
    
    # Confidence-based recommendations
    if quality_results.metrics.avg_entity_confidence < 0.6:
        recommendations.append("Low extraction confidence - consider improving source document quality")
    
    # Completeness-based recommendations
    if quality_results.metrics.property_completeness_rate < 60:
        recommendations.append("Low property completeness - review required property definitions")
    
    # Entity distribution recommendations
    if len(quality_results.metrics.entity_type_coverage) < 2:
        recommendations.append("Limited entity type diversity - verify document content matches schema")
    
    return recommendations