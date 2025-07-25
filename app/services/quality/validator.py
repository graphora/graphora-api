"""Main quality validation engine."""

import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

from app.services.transform.models import BaseNode, RelationshipInstance, DocumentKnowledgeGraph
from app.utils.constants import SYSTEM_PROPERTIES

from .models import (
    QualityResults, QualityViolation, QualityMetrics, ValidationResult,
    QualityRuleConfig, QualityRuleType, QualitySeverity
)
from .rules import QualityRuleFactory, QualityRule
from .ontology_parser import OntologyQualityParser

logger = logging.getLogger(__name__)


class QualityValidator:
    """Main quality validation engine that validates extracted data against ontology rules."""
    
    def __init__(self, ontology_with_rules: Dict[str, Any]):
        """Initialize validator with ontology that includes quality rules."""
        self.ontology = ontology_with_rules
        self.parser = OntologyQualityParser(ontology_with_rules)
        self.entity_rules = self._build_entity_rules()
        self.relationship_rules = self._build_relationship_rules()
        self.global_rules = self._build_global_rules()
        
        logger.info(
            f"Quality validator initialized with {len(self.entity_rules)} entity rule sets, "
            f"{len(self.relationship_rules)} relationship rule sets, "
            f"{len(self.global_rules)} global rules"
        )
        
    def _build_entity_rules(self) -> Dict[str, Dict[str, List[QualityRule]]]:
        """Build entity-level quality rules from ontology."""
        entity_rules = defaultdict(lambda: defaultdict(list))
        
        for entity_type, entity_def in self.ontology.get('entities', {}).items():
            logger.debug(f"Processing entity type: {entity_type}")
            # Property-level rules
            for prop_name, prop_def in entity_def.get('properties', {}).items():
                quality_config = prop_def.get('quality', {})
                if quality_config:
                    rules = self.parser.parse_property_quality_rules(
                        entity_type, prop_name, quality_config
                    )
                    entity_rules[entity_type][prop_name].extend(rules)
                    logger.debug(f"Added {len(rules)} quality rules for {entity_type}.{prop_name}")
                
                # Add required property rule if property is required
                if prop_def.get('required', False):
                    required_rule = self._create_required_property_rule(entity_type, prop_name)
                    entity_rules[entity_type][prop_name].append(required_rule)
                    logger.debug(f"Added required rule for {entity_type}.{prop_name}")
            
            # Entity-level rules
            entity_quality = entity_def.get('quality', {})
            if entity_quality:
                rules = self.parser.parse_entity_quality_rules(entity_type, entity_quality)
                entity_rules[entity_type]['_entityLevel'].extend(rules)
                logger.debug(f"Added {len(rules)} entity-level rules for {entity_type}")
        
        logger.debug(f"Built entity rules for {len(entity_rules)} entity types")
        return dict(entity_rules)
    
    def _build_relationship_rules(self) -> Dict[str, List[QualityRule]]:
        """Build relationship-level quality rules from ontology."""
        relationship_rules = defaultdict(list)
        
        for entity_type, entity_def in self.ontology.get('entities', {}).items():
            for rel_name, rel_def in entity_def.get('relationships', {}).items():
                quality_config = rel_def.get('quality', {})
                if quality_config:
                    rules = self.parser.parse_relationship_quality_rules(
                        rel_name, quality_config
                    )
                    relationship_rules[rel_name].extend(rules)
        
        return dict(relationship_rules)
    
    def _build_global_rules(self) -> List[QualityRule]:
        """Build global/cross-entity quality rules from ontology."""
        global_rules = []
        
        global_quality = self.ontology.get('dataQualityConfig', {})
        if global_quality:
            rules = self.parser.parse_global_quality_rules(global_quality)
            global_rules.extend(rules)
        
        return global_rules
    
    def _create_required_property_rule(self, entity_type: str, prop_name: str) -> QualityRule:
        """Create a required property rule."""
        config = QualityRuleConfig(
            rule_id=f"{entity_type}.{prop_name}.required",
            rule_type=QualityRuleType.BUSINESS,
            severity=QualitySeverity.ERROR,
            name=f"Required property: {prop_name}",
            description=f"Property {prop_name} is required for {entity_type}",
            parameters={'rule_class': 'required'}
        )
        return QualityRuleFactory.create_rule(config)
    
    async def validate_extraction(
        self, 
        knowledge_graph: DocumentKnowledgeGraph,
        transform_id: str
    ) -> QualityResults:
        """Validate extracted knowledge graph against quality rules."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting quality validation for transform {transform_id}")
            
            # Validate entities
            entity_violations = await self._validate_entities(knowledge_graph.nodes)
            
            # Validate relationships  
            relationship_violations = await self._validate_relationships(knowledge_graph.relationships)
            
            # Validate global rules (cross-entity constraints)
            global_violations = await self._validate_global_rules(knowledge_graph)
            
            # Combine all violations
            all_violations = entity_violations + relationship_violations + global_violations
            
            # Calculate metrics
            metrics = self._calculate_metrics(knowledge_graph, all_violations)
            
            # Calculate overall score and grade
            overall_score = self._calculate_overall_score(metrics, all_violations)
            grade = self._calculate_grade(overall_score)
            
            # Determine if human review is required
            requires_review = self._requires_human_review(all_violations, overall_score)
            
            # Build summary statistics
            violations_by_type = self._group_violations_by_type(all_violations)
            violations_by_severity = self._group_violations_by_severity(all_violations)
            violations_by_entity_type = self._group_violations_by_entity_type(all_violations)
            entity_quality_summary = self._build_entity_quality_summary(all_violations)
            
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            
            results = QualityResults(
                transform_id=transform_id,
                overall_score=overall_score,
                grade=grade,
                requires_review=requires_review,
                violations=all_violations,
                metrics=metrics,
                violations_by_type=violations_by_type,
                violations_by_severity=violations_by_severity,
                violations_by_entity_type=violations_by_entity_type,
                entity_quality_summary=entity_quality_summary,
                validation_duration_ms=duration_ms,
                rules_applied=len(self._get_all_rules()),
                validation_config={'ontology_version': self.ontology.get('version', 'unknown')}
            )
            
            logger.info(
                f"Quality validation completed for {transform_id}: "
                f"Score={overall_score:.1f}, Grade={grade}, "
                f"Violations={len(all_violations)}, Duration={duration_ms}ms"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Quality validation failed for transform {transform_id}: {e}")
            raise
    
    async def _validate_entities(self, entities: List[BaseNode]) -> List[QualityViolation]:
        """Validate all entities against entity-level quality rules."""
        violations = []
        
        for entity in entities:
            entity_type = entity.type
            
            # Skip entities with no rules defined
            if entity_type not in self.entity_rules:
                continue
            
            # Validate each property
            for prop_name, prop_value in entity.properties.items():
                # Skip system properties
                if prop_name in SYSTEM_PROPERTIES:
                    continue
                
                # Get rules for this property
                prop_rules = self.entity_rules[entity_type].get(prop_name, [])
                
                for rule in prop_rules:
                    if not rule.is_enabled():
                        continue
                    
                    context = {
                        'entity_type': entity_type,
                        'entity_id': entity.id,
                        'property_name': prop_name,
                        'entity': entity
                    }
                    
                    try:
                        result = rule.validate(prop_value, context)
                        violations.extend(result.violations)
                    except Exception as e:
                        logger.error(f"Rule {rule.rule_id} failed on {entity_type}.{prop_name}: {e}")
                        # Create a violation for the rule failure itself
                        violation = QualityViolation(
                            rule_id=rule.rule_id,
                            rule_type=rule.rule_type,
                            severity=QualitySeverity.ERROR,
                            entity_type=entity_type,
                            entity_id=entity.id,
                            property_name=prop_name,
                            message=f"Quality rule execution failed: {str(e)}",
                            expected="Rule should execute successfully",
                            actual="Rule execution error",
                            confidence=1.0
                        )
                        violations.append(violation)
            
            # Validate entity-level rules
            entity_level_rules = self.entity_rules[entity_type].get('_entityLevel', [])
            for rule in entity_level_rules:
                if not rule.is_enabled():
                    continue
                
                context = {
                    'entity_type': entity_type,
                    'entity_id': entity.id,
                    'entity': entity
                }
                
                try:
                    result = rule.validate(entity, context)
                    violations.extend(result.violations)
                except Exception as e:
                    logger.error(f"Entity-level rule {rule.rule_id} failed on {entity_type}: {e}")
        
        return violations
    
    async def _validate_relationships(self, relationships: List[RelationshipInstance]) -> List[QualityViolation]:
        """Validate all relationships against relationship-level quality rules."""
        violations = []
        
        for relationship in relationships:
            rel_type = relationship.type
            
            # Skip relationships with no rules defined
            if rel_type not in self.relationship_rules:
                continue
            
            rules = self.relationship_rules[rel_type]
            
            for rule in rules:
                if not rule.is_enabled():
                    continue
                
                context = {
                    'relationship_type': rel_type,
                    'source_id': relationship.source_id,
                    'target_id': relationship.target_id,
                    'relationship': relationship
                }
                
                try:
                    result = rule.validate(relationship, context)
                    violations.extend(result.violations)
                except Exception as e:
                    logger.error(f"Relationship rule {rule.rule_id} failed on {rel_type}: {e}")
        
        return violations
    
    async def _validate_global_rules(self, knowledge_graph: DocumentKnowledgeGraph) -> List[QualityViolation]:
        """Validate global/cross-entity rules."""
        violations = []
        
        for rule in self.global_rules:
            if not rule.is_enabled():
                continue
            
            context = {
                'knowledge_graph': knowledge_graph,
                'entities': knowledge_graph.nodes,
                'relationships': knowledge_graph.relationships
            }
            
            try:
                result = rule.validate(knowledge_graph, context)
                violations.extend(result.violations)
            except Exception as e:
                logger.error(f"Global rule {rule.rule_id} failed: {e}")
        
        return violations
    
    def _calculate_metrics(self, kg: DocumentKnowledgeGraph, violations: List[QualityViolation]) -> QualityMetrics:
        """Calculate quality metrics from the knowledge graph and violations."""
        total_entities = len(kg.nodes)
        total_relationships = len(kg.relationships)
        total_properties = sum(len(entity.properties) for entity in kg.nodes)
        
        # Count entities and relationships with violations
        entities_with_violations = len(set(v.entity_id for v in violations if v.entity_id))
        relationships_with_violations = len(set(
            f"{v.context.get('source_id', '')}-{v.relationship_type}-{v.context.get('target_id', '')}"
            for v in violations if v.relationship_type
        ))
        
        # Calculate rates
        entity_violation_rate = (entities_with_violations / max(total_entities, 1)) * 100
        relationship_violation_rate = (relationships_with_violations / max(total_relationships, 1)) * 100
        overall_violation_rate = (len(violations) / max(total_entities + total_relationships, 1)) * 100
        
        # Calculate confidence scores
        entity_confidences = [entity.confidence_score for entity in kg.nodes if entity.confidence_score is not None]
        avg_entity_confidence = sum(entity_confidences) / max(len(entity_confidences), 1)
        
        # Group confidence by entity type
        confidence_by_type = defaultdict(list)
        for entity in kg.nodes:
            if entity.confidence_score is not None:
                confidence_by_type[entity.type].append(entity.confidence_score)
        
        confidence_scores_by_type = {
            entity_type: sum(scores) / len(scores)
            for entity_type, scores in confidence_by_type.items()
        }
        
        # Calculate property completeness (non-system properties)
        # Use both required properties and all defined properties for a more complete picture
        required_properties = 0
        filled_required_properties = 0
        total_expected_properties = 0
        total_filled_properties = 0
        
        for entity_type, entity_def in self.ontology.get('entities', {}).items():
            entities_of_type = [e for e in kg.nodes if e.type == entity_type]
            entity_count = len(entities_of_type)
            
            for prop_name, prop_def in entity_def.get('properties', {}).items():
                if prop_name in SYSTEM_PROPERTIES:
                    continue
                
                # Count all defined properties (for overall completeness)
                total_expected_properties += entity_count
                filled_count = len([
                    e for e in entities_of_type
                    if prop_name in e.properties and 
                    e.properties[prop_name] is not None and
                    str(e.properties[prop_name]).strip() != ""
                ])
                total_filled_properties += filled_count
                
                # Also track required properties specifically
                if prop_def.get('required', False):
                    required_properties += entity_count
                    filled_required_properties += filled_count
                    logger.debug(
                        f"Required property: {entity_type}.{prop_name} - "
                        f"entities: {entity_count}, filled: {filled_count}"
                    )
        
        # Use required properties if available, otherwise use all properties
        if required_properties > 0:
            property_completeness_rate = (filled_required_properties / required_properties) * 100
            logger.info(
                f"Property completeness (required): {filled_required_properties}/{required_properties} = {property_completeness_rate:.1f}%"
            )
        elif total_expected_properties > 0:
            property_completeness_rate = (total_filled_properties / total_expected_properties) * 100
            logger.info(
                f"Property completeness (all): {total_filled_properties}/{total_expected_properties} = {property_completeness_rate:.1f}%"
            )
        else:
            property_completeness_rate = 100.0  # No properties defined, consider complete
            logger.info("No properties defined in ontology, setting completeness to 100%")
        
        # Entity type coverage
        entity_type_coverage = defaultdict(int)
        for entity in kg.nodes:
            entity_type_coverage[entity.type] += 1
        
        return QualityMetrics(
            total_entities=total_entities,
            total_relationships=total_relationships,
            total_properties=total_properties,
            entities_with_violations=entities_with_violations,
            relationships_with_violations=relationships_with_violations,
            total_violations=len(violations),
            entity_violation_rate=entity_violation_rate,
            relationship_violation_rate=relationship_violation_rate,
            overall_violation_rate=overall_violation_rate,
            avg_entity_confidence=avg_entity_confidence,
            avg_relationship_confidence=0.0,  # TODO: Add relationship confidence if available
            confidence_scores_by_type=dict(confidence_scores_by_type),
            property_completeness_rate=property_completeness_rate,
            entity_type_coverage=dict(entity_type_coverage)
        )
    
    def _calculate_overall_score(self, metrics: QualityMetrics, violations: List[QualityViolation]) -> float:
        """Calculate overall quality score (0-100) based on metrics and violations."""
        base_score = 100.0
        
        # Penalize based on violation severity
        error_penalty = len([v for v in violations if v.severity == QualitySeverity.ERROR]) * 10
        warning_penalty = len([v for v in violations if v.severity == QualitySeverity.WARNING]) * 5
        info_penalty = len([v for v in violations if v.severity == QualitySeverity.INFO]) * 1
        
        penalty = error_penalty + warning_penalty + info_penalty
        
        # Factor in property completeness
        completeness_bonus = (metrics.property_completeness_rate - 50) * 0.2  # Bonus/penalty for completeness
        
        # Factor in confidence scores
        confidence_bonus = (metrics.avg_entity_confidence - 0.5) * 20  # Bonus/penalty for confidence
        
        final_score = base_score - penalty + completeness_bonus + confidence_bonus
        
        # Ensure score is within bounds
        return max(0.0, min(100.0, final_score))
    
    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _requires_human_review(self, violations: List[QualityViolation], score: float) -> bool:
        """Determine if human review is required based on violations and score."""
        # Require review if there are any errors
        has_errors = any(v.severity == QualitySeverity.ERROR for v in violations)
        if has_errors:
            return True
        
        # Require review if score is below threshold
        if score < 80.0:
            return True
        
        # Require review if there are too many warnings
        warning_count = len([v for v in violations if v.severity == QualitySeverity.WARNING])
        if warning_count > 10:
            return True
        
        return False
    
    def _group_violations_by_type(self, violations: List[QualityViolation]) -> Dict[QualityRuleType, int]:
        """Group violations by rule type."""
        groups = defaultdict(int)
        for violation in violations:
            groups[violation.rule_type] += 1
        return dict(groups)
    
    def _group_violations_by_severity(self, violations: List[QualityViolation]) -> Dict[QualitySeverity, int]:
        """Group violations by severity."""
        groups = defaultdict(int)
        for violation in violations:
            groups[violation.severity] += 1
        return dict(groups)
    
    def _group_violations_by_entity_type(self, violations: List[QualityViolation]) -> Dict[str, int]:
        """Group violations by entity type."""
        groups = defaultdict(int)
        for violation in violations:
            if violation.entity_type:
                groups[violation.entity_type] += 1
        return dict(groups)
    
    def _build_entity_quality_summary(self, violations: List[QualityViolation]) -> Dict[str, Dict[str, int]]:
        """Build entity quality summary: {entity_type: {severity: count}}."""
        summary = defaultdict(lambda: defaultdict(int))
        for violation in violations:
            if violation.entity_type:
                summary[violation.entity_type][violation.severity] += 1
        
        # Convert to regular dict
        return {
            entity_type: dict(severity_counts)
            for entity_type, severity_counts in summary.items()
        }
    
    def _get_all_rules(self) -> List[QualityRule]:
        """Get all rules for counting purposes."""
        all_rules = []
        
        # Entity rules
        entity_rule_count = 0
        for entity_type, prop_rules in self.entity_rules.items():
            for prop_name, rules in prop_rules.items():
                entity_rule_count += len(rules)
                all_rules.extend(rules)
        
        # Relationship rules
        relationship_rule_count = 0
        for rel_type, rules in self.relationship_rules.items():
            relationship_rule_count += len(rules)
            all_rules.extend(rules)
        
        # Global rules
        global_rule_count = len(self.global_rules)
        all_rules.extend(self.global_rules)
        
        logger.info(
            f"Total rules count: {len(all_rules)} "
            f"(Entity: {entity_rule_count}, Relationship: {relationship_rule_count}, Global: {global_rule_count})"
        )
        
        return all_rules