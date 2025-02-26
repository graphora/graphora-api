"""Unit tests for the LLM Conflict Analyzer."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.merge.llm_analyzer import LLMConflictAnalyzer
from app.schemas.conflicts import Conflict, ConflictType, ResolutionOption, ConflictSeverity


@pytest.fixture
def llm_analyzer():
    return LLMConflictAnalyzer()


@pytest.fixture
def mock_property_conflict():
    return Conflict(
        id="conflict1",
        conflict_type=ConflictType.PROPERTY,
        severity=ConflictSeverity.MAJOR,
        entity_id="entity1",
        property_name="age",
        staging_value="25",
        production_value="30",
        entity_type="Person",
        resolved=False,
        merge_id="merge1",
        description="",
        resolution=None,
        analysis=None
    )


@pytest.fixture
def mock_relationship_conflict():
    return Conflict(
        id="conflict2",
        conflict_type=ConflictType.RELATIONSHIP,
        severity=ConflictSeverity.MAJOR,
        relationship_id="rel1",
        staging_relationship_type="WORKS_FOR",
        production_relationship_type="EMPLOYED_BY",
        resolved=False,
        merge_id="merge2",
        description="",
        resolution=None,
        analysis=None
    )


@pytest.fixture
def mock_entity_match_conflict():
    return Conflict(
        id="conflict3",
        conflict_type=ConflictType.ENTITY_MATCH,
        severity=ConflictSeverity.MAJOR,
        source_entity_id="entity1",
        target_entity_id="entity2",
        similarity_score=0.9,
        resolved=False,
        merge_id="merge3",
        description="",
        resolution=None,
        analysis=None
    )


@pytest.fixture
def mock_ontology():
    return {
        "entity_types": ["Person", "Organization", "Product"],
        "relationship_types": ["WORKS_FOR", "OWNS", "RELATED_TO", "EMPLOYED_BY"],
        "property_constraints": {
            "age": {"type": "integer", "min": 0, "max": 120},
            "name": {"type": "string", "required": True},
            "email": {"type": "string", "format": "email"}
        }
    }


@pytest.mark.asyncio
async def test_analyze_property_conflict(llm_analyzer, mock_property_conflict, mock_ontology):
    """Test analyzing property value conflict."""
    with patch.object(llm_analyzer, '_analyze_with_llm', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = [
            {
                "value": "30",
                "confidence": 0.8,
                "explanation": "The target value is likely correct based on other records."
            },
            {
                "value": "25",
                "confidence": 0.2,
                "explanation": "The source value appears to be outdated."
            }
        ]
        
        options = await llm_analyzer.analyze_property_conflict(
            mock_property_conflict, 
            mock_ontology
        )
        
        assert len(options) == 2
        assert options[0].confidence == 0.8
        assert "target value is likely correct" in options[0].reasoning


@pytest.mark.asyncio
async def test_analyze_relationship_conflict(llm_analyzer, mock_relationship_conflict, mock_ontology):
    """Test analyzing relationship type conflict."""
    with patch.object(llm_analyzer, '_analyze_with_llm', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = [
            {
                "type": "EMPLOYED_BY",
                "confidence": 0.6,
                "explanation": "This is the standard term in the ontology."
            },
            {
                "type": "WORKS_FOR",
                "confidence": 0.4,
                "explanation": "This is a common synonym but less precise."
            }
        ]
        
        options = await llm_analyzer.analyze_relationship_conflict(
            mock_relationship_conflict, 
            mock_ontology
        )
        
        assert len(options) == 2
        assert options[0].confidence == 0.6
        assert "standard term" in options[0].reasoning


@pytest.mark.asyncio
async def test_analyze_entity_match_conflict(llm_analyzer, mock_entity_match_conflict, mock_ontology):
    """Test analyzing duplicate entity conflict."""
    with patch.object(llm_analyzer, '_analyze_with_llm', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = [
            {
                "action": "merge",
                "confidence": 0.9,
                "explanation": "Entities appear to represent the same real-world object."
            },
            {
                "action": "keep_separate",
                "confidence": 0.1,
                "explanation": "Although similar, these may be distinct entities."
            }
        ]
        
        options = await llm_analyzer.analyze_entity_match_conflict(
            mock_entity_match_conflict, 
            mock_ontology
        )
        
        assert len(options) == 2
        assert options[0].confidence == 0.9
        assert "same real-world object" in options[0].reasoning


@pytest.mark.asyncio
async def test_analyze_conflict_handles_all_types(llm_analyzer, mock_property_conflict, mock_relationship_conflict, mock_entity_match_conflict, mock_ontology):
    """Test that analyze_conflict correctly handles all conflict types."""
    # Mock all analysis methods
    with patch.object(llm_analyzer, 'analyze_property_conflict', new_callable=AsyncMock) as mock_property_analyzer, \
         patch.object(llm_analyzer, 'analyze_relationship_conflict', new_callable=AsyncMock) as mock_relationship_analyzer, \
         patch.object(llm_analyzer, 'analyze_entity_match_conflict', new_callable=AsyncMock) as mock_duplicate_analyzer:
        
        # Set return values
        property_option = ResolutionOption(
            id="test1",
            description="Test option",
            resolution_type="KEEP_STAGING",
            resolution_data={},
            confidence=0.8,
            reasoning="Test reasoning",
            requires_review=False,
            auto_resolvable=True
        )
        relationship_option = ResolutionOption(
            id="test2",
            description="Test option",
            resolution_type="KEEP_STAGING_REL",
            resolution_data={},
            confidence=0.7,
            reasoning="Test reasoning",
            requires_review=False,
            auto_resolvable=True
        )
        entity_option = ResolutionOption(
            id="test3",
            description="Test option",
            resolution_type="MERGE_VALUES",
            resolution_data={},
            confidence=0.9,
            reasoning="Test reasoning",
            requires_review=False,
            auto_resolvable=True
        )
        
        mock_property_analyzer.return_value = [property_option]
        mock_relationship_analyzer.return_value = [relationship_option]
        mock_duplicate_analyzer.return_value = [entity_option]
        
        # Test each conflict type
        property_result = await llm_analyzer.analyze_conflict(mock_property_conflict, mock_ontology)
        assert property_result == [property_option]
        mock_property_analyzer.assert_called_once()
        
        relationship_result = await llm_analyzer.analyze_conflict(mock_relationship_conflict, mock_ontology)
        assert relationship_result == [relationship_option]
        mock_relationship_analyzer.assert_called_once()
        
        entity_result = await llm_analyzer.analyze_conflict(mock_entity_match_conflict, mock_ontology)
        assert entity_result == [entity_option]
        mock_duplicate_analyzer.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_conflict_fallback_to_defaults(llm_analyzer, mock_property_conflict, mock_ontology):
    """Test that analyze_conflict falls back to default options when analysis fails."""
    with patch.object(llm_analyzer, 'analyze_property_conflict', new_callable=AsyncMock) as mock_analyzer:
        # Simulate analysis failure
        mock_analyzer.side_effect = Exception("LLM analysis failed")
        
        # Test that default options are returned
        result = await llm_analyzer.analyze_conflict(mock_property_conflict, mock_ontology)
        
        # Verify default options were returned
        assert len(result) == 2
        assert all(isinstance(option, ResolutionOption) for option in result)
        assert any(option.resolution_type == "keep_staging" for option in result)
        assert any(option.resolution_type == "keep_production" for option in result)
