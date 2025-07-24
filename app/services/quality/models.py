"""Quality validation models and data structures."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime


class QualityRuleType(str, Enum):
    """Types of quality rules that can be applied."""
    FORMAT = "format"
    BUSINESS = "business" 
    CROSS_ENTITY = "cross_entity"
    DISTRIBUTION = "distribution"
    CONSISTENCY = "consistency"


class QualitySeverity(str, Enum):
    """Severity levels for quality violations."""
    ERROR = "error"       # Critical issues that should block processing
    WARNING = "warning"   # Issues that should be reviewed but may be acceptable
    INFO = "info"         # Informational notices


class QualityViolation(BaseModel):
    """Represents a single quality rule violation."""
    rule_id: str = Field(description="Unique identifier for the rule that was violated")
    rule_type: QualityRuleType = Field(description="Type of quality rule")
    severity: QualitySeverity = Field(description="Severity level of the violation")
    
    # Context information
    entity_type: Optional[str] = Field(None, description="Type of entity where violation occurred")
    entity_id: Optional[str] = Field(None, description="ID of the specific entity")
    property_name: Optional[str] = Field(None, description="Property name where violation occurred")
    relationship_type: Optional[str] = Field(None, description="Type of relationship if applicable")
    
    # Violation details  
    message: str = Field(description="Human-readable description of the violation")
    expected: str = Field(description="What was expected")
    actual: str = Field(description="What was actually found")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this violation detection")
    
    # Suggestions for fixing
    suggestion: Optional[str] = Field(None, description="Suggested fix for the violation")
    
    # Additional context
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the violation")


class QualityMetrics(BaseModel):
    """Overall quality metrics for an extraction."""
    total_entities: int = Field(description="Total number of entities extracted")
    total_relationships: int = Field(description="Total number of relationships extracted")
    total_properties: int = Field(description="Total number of properties across all entities")
    
    # Violation statistics
    entities_with_violations: int = Field(description="Number of entities that have violations")
    relationships_with_violations: int = Field(description="Number of relationships that have violations")
    total_violations: int = Field(description="Total number of violations found")
    
    # Rates and percentages  
    entity_violation_rate: float = Field(description="Percentage of entities with violations")
    relationship_violation_rate: float = Field(description="Percentage of relationships with violations")
    overall_violation_rate: float = Field(description="Overall violation rate")
    
    # Confidence scores
    avg_entity_confidence: float = Field(description="Average confidence score for entity extraction")
    avg_relationship_confidence: float = Field(description="Average confidence score for relationship extraction")
    confidence_scores_by_type: Dict[str, float] = Field(default_factory=dict, description="Average confidence by entity type")
    
    # Completeness metrics
    property_completeness_rate: float = Field(description="Percentage of required properties that were filled")
    entity_type_coverage: Dict[str, int] = Field(default_factory=dict, description="Count of entities by type")


class QualityResults(BaseModel):
    """Complete quality validation results for a transform."""
    transform_id: str = Field(description="ID of the transform that was validated")
    overall_score: float = Field(ge=0.0, le=100.0, description="Overall quality score (0-100)")
    grade: str = Field(description="Letter grade (A, B, C, D, F)")
    requires_review: bool = Field(description="Whether human review is required")
    
    # Detailed results
    violations: List[QualityViolation] = Field(default_factory=list, description="List of all violations found")
    metrics: QualityMetrics = Field(description="Overall quality metrics")
    
    # Summary breakdowns
    violations_by_type: Dict[QualityRuleType, int] = Field(default_factory=dict, description="Violation count by rule type")
    violations_by_severity: Dict[QualitySeverity, int] = Field(default_factory=dict, description="Violation count by severity")
    violations_by_entity_type: Dict[str, int] = Field(default_factory=dict, description="Violation count by entity type")
    
    # Entity-level summary: {entity_type: {severity: count}}
    entity_quality_summary: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Quality summary by entity type")
    
    # Validation metadata
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When validation was performed")
    validation_duration_ms: int = Field(description="How long validation took in milliseconds")
    rules_applied: int = Field(description="Number of quality rules that were applied")
    
    # Configuration used
    validation_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration used for validation")


class ValidationResult(BaseModel):
    """Result of applying a single quality rule."""
    is_valid: bool = Field(description="Whether the validation passed")
    message: str = Field(description="Validation message")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the validation result")
    violations: List[QualityViolation] = Field(default_factory=list, description="Any violations found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QualityRuleConfig(BaseModel):
    """Configuration for a quality rule."""
    rule_id: str = Field(description="Unique identifier for the rule")
    rule_type: QualityRuleType = Field(description="Type of quality rule")
    severity: QualitySeverity = Field(default=QualitySeverity.WARNING, description="Severity of violations")
    enabled: bool = Field(default=True, description="Whether the rule is enabled")
    
    # Rule parameters (flexible for different rule types)
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule-specific parameters")
    
    # Metadata
    name: str = Field(description="Human-readable name for the rule")
    description: str = Field(description="Description of what the rule validates")
    examples: List[str] = Field(default_factory=list, description="Examples of valid/invalid values")


# Type aliases for commonly used types
QualityScore = float  # 0.0 to 100.0
ConfidenceScore = float  # 0.0 to 1.0
EntityType = str
PropertyName = str
RuleID = str