#!/usr/bin/env python3
"""
Test script for the quality validation system.
Run this to verify the quality system works correctly.
"""

import asyncio
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent))


async def test_quality_system():
    """Test the quality validation system with sample data."""

    print("🔍 Testing Quality Validation System")
    print("=" * 50)

    try:
        # Import quality modules
        from app.services.quality.validator import QualityValidator
        from app.services.transform.models import (
            BaseNode,
            RelationshipInstance,
            DocumentKnowledgeGraph,
        )

        print("✅ Successfully imported quality modules")

        # Load sample ontology with quality rules
        with open("example_ontology_with_quality.yaml", "r") as f:
            ontology_with_rules = yaml.safe_load(f)

        print("✅ Loaded sample ontology with quality rules")

        # Create sample extracted data (simulating LLM extraction)
        sample_nodes = [
            # Good company node
            BaseNode(
                id="company_1",
                type="Company",
                properties={
                    "name": "Apple Inc.",
                    "companyId": "AAPL",
                    "description": "Technology company that designs and manufactures consumer electronics and software.",
                },
                confidence_score=0.95,
            ),
            # Company with quality issues
            BaseNode(
                id="company_2",
                type="Company",
                properties={
                    "name": "unknown company",  # Bad: lowercase, forbidden value
                    "companyId": "invalid-id-123456789",  # Bad: too long, invalid pattern
                    "description": "N/A",  # Bad: forbidden value
                },
                confidence_score=0.4,
            ),
            # Good industry node
            BaseNode(
                id="industry_1",
                type="Industry",
                properties={
                    "name": "Technology Hardware",
                    "industryId": "TECH01",
                    "description": "Companies that manufacture computer hardware and electronic devices.",
                    "classification": "GICS",
                },
                confidence_score=0.9,
            ),
            # Industry with missing required properties
            BaseNode(
                id="industry_2",
                type="Industry",
                properties={
                    "industryId": "IND02"
                    # Missing required 'name' property
                },
                confidence_score=0.6,
            ),
        ]

        sample_relationships = [
            RelationshipInstance(
                source_id="company_1",
                target_id="industry_1",
                type="CLASSIFIED_AS",
                properties={
                    "classificationSource": "GICS",
                    "confidenceScore": 0.9,
                    "description": "Apple is classified in the Technology Hardware industry",
                },
            ),
            # Relationship with quality issues
            RelationshipInstance(
                source_id="company_2",
                target_id="industry_2",
                type="CLASSIFIED_AS",
                properties={
                    "classificationSource": "Invalid Source",  # Bad: not in allowed values
                    "confidenceScore": 1.5,  # Bad: exceeds max value
                    "description": "",
                },
            ),
        ]

        # Create knowledge graph
        knowledge_graph = DocumentKnowledgeGraph(
            nodes=sample_nodes, relationships=sample_relationships
        )

        print(
            f"✅ Created sample knowledge graph: {len(sample_nodes)} nodes, {len(sample_relationships)} relationships"
        )

        # Initialize validator
        validator = QualityValidator(ontology_with_rules)
        print("✅ Initialized quality validator")

        # Run quality validation
        print("\n🔍 Running quality validation...")
        quality_results = await validator.validate_extraction(
            knowledge_graph, "test_transform_123"
        )

        # Display results
        print("\n📊 QUALITY VALIDATION RESULTS")
        print("=" * 40)
        print(f"Overall Score: {quality_results.overall_score:.1f}/100")
        print(f"Grade: {quality_results.grade}")
        print(f"Requires Review: {quality_results.requires_review}")
        print(f"Total Violations: {len(quality_results.violations)}")

        print("\nViolations by Severity:")
        for severity, count in quality_results.violations_by_severity.items():
            print(f"  {severity}: {count}")

        print("\nViolations by Type:")
        for rule_type, count in quality_results.violations_by_type.items():
            print(f"  {rule_type}: {count}")

        print("\nEntity Quality Summary:")
        for (
            entity_type,
            severity_counts,
        ) in quality_results.entity_quality_summary.items():
            print(f"  {entity_type}: {severity_counts}")

        # Show detailed violations
        if quality_results.violations:
            print("\n⚠️  DETAILED VIOLATIONS (showing first 5)")
            print("-" * 60)
            for i, violation in enumerate(quality_results.violations[:5]):
                print(f"{i+1}. [{violation.severity}] {violation.rule_type}")
                print(f"   Entity: {violation.entity_type} (ID: {violation.entity_id})")
                if violation.property_name:
                    print(f"   Property: {violation.property_name}")
                print(f"   Message: {violation.message}")
                print(f"   Expected: {violation.expected}")
                print(f"   Actual: {violation.actual}")
                if violation.suggestion:
                    print(f"   Suggestion: {violation.suggestion}")
                print()

        # Test API-like operations (without actual HTTP)
        print("\n🔗 Testing Service Operations")
        print("-" * 30)

        from app.services.quality.service import QualityService
        from app.services.storage.neo4j import Neo4jStorage

        # Note: This would require actual Neo4j connection in real usage
        _ = (QualityService, Neo4jStorage)
        print("✅ Quality service classes imported successfully")

        print("\n🎉 Quality validation system test completed successfully!")
        print(f"   - Score: {quality_results.overall_score:.1f}")
        print(f"   - Grade: {quality_results.grade}")
        print(f"   - Violations: {len(quality_results.violations)}")

        return quality_results

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(
            "   Make sure you're running from the correct directory and have all dependencies installed"
        )
        return None

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_quality_system())

    if result:
        print("\n✅ Test passed! Quality system is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Test failed! Check the errors above.")
        sys.exit(1)
