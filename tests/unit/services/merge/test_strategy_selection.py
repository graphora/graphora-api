"""Unit tests for the strategy selection system"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import uuid
from datetime import datetime, timedelta

from app.services.merge.strategy_selection import (
    ResolutionStrategy,
    PreferStagingStrategy,
    PreferProductionStrategy,
    MergeValuesStrategy,
    KeepBothStrategy,
    LLMBasedStrategy,
    StrategySelectionEngine
)
from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionOption,
    StrategyType
)

@pytest.fixture
def property_value_conflict():
    """Create a sample property value conflict"""
    return Conflict(
        id=str(uuid.uuid4()),
        merge_id=str(uuid.uuid4()),
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MINOR,
        entity_id="node123",
        entity_type="Person",
        property_name="name",
        staging_value="John Smith",
        production_value="john smith",
        description="Case difference in name property",
        context={
            "staging_timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "production_timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
        },
        resolution_options=[
            ResolutionOption(
                id="opt1",
                description="Keep staging value 'John Smith'",
                resolution_type="keep_staging_value",
                confidence=0.8,
                auto_resolvable=True
            ),
            ResolutionOption(
                id="opt2",
                description="Keep production value 'john smith'",
                resolution_type="keep_production_value",
                confidence=0.6,
                auto_resolvable=True
            )
        ]
    )

@pytest.fixture
def relationship_conflict():
    """Create a sample relationship conflict"""
    return Conflict(
        id=str(uuid.uuid4()),
        merge_id=str(uuid.uuid4()),
        conflict_type=ConflictType.RELATIONSHIP_TYPE,
        severity=ConflictSeverity.MAJOR,
        entity_id="rel456",
        entity_type="WORKS_FOR",
        description="Different relationship types",
        context={
            "staging_type": "WORKS_FOR",
            "production_type": "EMPLOYED_BY"
        },
        resolution_options=[
            ResolutionOption(
                id="opt1",
                description="Keep staging relationship type 'WORKS_FOR'",
                resolution_type="keep_staging_rel",
                confidence=0.7,
                auto_resolvable=True
            ),
            ResolutionOption(
                id="opt2",
                description="Keep production relationship type 'EMPLOYED_BY'",
                resolution_type="keep_production_rel",
                confidence=0.6,
                auto_resolvable=True
            ),
            ResolutionOption(
                id="opt3",
                description="Keep both relationships",
                resolution_type="keep_both_relationships",
                confidence=0.5,
                auto_resolvable=True
            )
        ]
    )

@pytest.fixture
def list_property_conflict():
    """Create a sample conflict with list properties"""
    return Conflict(
        id=str(uuid.uuid4()),
        merge_id=str(uuid.uuid4()),
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MINOR,
        entity_id="node789",
        entity_type="Product",
        property_name="tags",
        staging_value=["electronics", "computer", "laptop"],
        production_value=["electronics", "hardware", "computer"],
        description="Different tags for product",
        context={
            "staging_value": ["electronics", "computer", "laptop"],
            "production_value": ["electronics", "hardware", "computer"]
        },
        resolution_options=[
            ResolutionOption(
                id="opt1",
                description="Keep staging tags",
                resolution_type="keep_staging_value",
                confidence=0.6,
                auto_resolvable=True
            ),
            ResolutionOption(
                id="opt2",
                description="Keep production tags",
                resolution_type="keep_production_value",
                confidence=0.5,
                auto_resolvable=True
            ),
            ResolutionOption(
                id="opt3",
                description="Merge all unique tags",
                resolution_type="merge_values",
                confidence=0.8,
                auto_resolvable=True
            )
        ]
    )

@pytest.fixture
def critical_conflict():
    """Create a sample critical conflict"""
    return Conflict(
        id=str(uuid.uuid4()),
        merge_id=str(uuid.uuid4()),
        conflict_type=ConflictType.ENTITY_MATCH,
        severity=ConflictSeverity.CRITICAL,
        entity_id="node999",
        entity_type="Organization",
        description="Potential duplicate organization entities",
        context={
            "staging_entity": {"id": "org1", "name": "Acme Inc."},
            "production_candidates": [
                {"id": "org2", "name": "ACME Incorporated", "match_score": 0.85},
                {"id": "org3", "name": "Acme International", "match_score": 0.75}
            ]
        },
        resolution_options=[
            ResolutionOption(
                id="opt1",
                description="Match with 'ACME Incorporated'",
                resolution_type="match_entity",
                resolution_data={"production_id": "org2"},
                confidence=0.85,
                auto_resolvable=False
            ),
            ResolutionOption(
                id="opt2",
                description="Match with 'Acme International'",
                resolution_type="match_entity",
                resolution_data={"production_id": "org3"},
                confidence=0.75,
                auto_resolvable=False
            ),
            ResolutionOption(
                id="opt3",
                description="Create new entity",
                resolution_type="create_new",
                confidence=0.6,
                auto_resolvable=False
            )
        ]
    )

class TestPreferStagingStrategy:
    """Tests for the PreferStagingStrategy"""
    
    def test_get_confidence_property_value(self, property_value_conflict):
        """Test confidence calculation for property value conflicts"""
        strategy = PreferStagingStrategy()
        confidence = strategy.get_confidence(property_value_conflict)
        
        # Should have high confidence due to staging being newer
        assert confidence > 0.7
        
    def test_get_resolution_option(self, property_value_conflict):
        """Test getting the correct resolution option"""
        strategy = PreferStagingStrategy()
        option = strategy.get_resolution_option(property_value_conflict)
        
        assert option is not None
        assert option.id == "opt1"
        assert option.resolution_type == "keep_staging_value"

class TestPreferProductionStrategy:
    """Tests for the PreferProductionStrategy"""
    
    def test_get_confidence_id_property(self):
        """Test confidence calculation for ID properties"""
        conflict = Conflict(
            id=str(uuid.uuid4()),
            merge_id=str(uuid.uuid4()),
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            entity_id="node123",
            entity_type="Person",
            property_name="employee_id",
            staging_value="EMP-123",
            production_value="EMP-456",
            description="Different employee IDs",
            resolution_options=[
                ResolutionOption(
                    id="opt1",
                    description="Keep staging value",
                    resolution_type="keep_staging_value",
                    confidence=0.5,
                    auto_resolvable=True
                ),
                ResolutionOption(
                    id="opt2",
                    description="Keep production value",
                    resolution_type="keep_production_value",
                    confidence=0.5,
                    auto_resolvable=True
                )
            ]
        )
        
        strategy = PreferProductionStrategy()
        confidence = strategy.get_confidence(conflict)
        
        # Should have high confidence for ID fields
        assert confidence >= 0.8
        
    def test_get_resolution_option(self, property_value_conflict):
        """Test getting the correct resolution option"""
        strategy = PreferProductionStrategy()
        option = strategy.get_resolution_option(property_value_conflict)
        
        assert option is not None
        assert option.id == "opt2"
        assert option.resolution_type == "keep_production_value"

class TestMergeValuesStrategy:
    """Tests for the MergeValuesStrategy"""
    
    def test_get_confidence_list_property(self, list_property_conflict):
        """Test confidence calculation for list properties"""
        strategy = MergeValuesStrategy()
        confidence = strategy.get_confidence(list_property_conflict)
        
        # Should have high confidence for list properties
        assert confidence >= 0.7
        
    def test_get_confidence_non_mergeable(self, property_value_conflict):
        """Test confidence calculation for non-mergeable properties"""
        strategy = MergeValuesStrategy()
        confidence = strategy.get_confidence(property_value_conflict)
        
        # Should have low confidence for non-mergeable properties
        assert confidence < 0.6
        
    def test_get_resolution_option(self, list_property_conflict):
        """Test getting the correct resolution option"""
        strategy = MergeValuesStrategy()
        option = strategy.get_resolution_option(list_property_conflict)
        
        assert option is not None
        assert option.id == "opt3"
        assert option.resolution_type == "merge_values"

class TestKeepBothStrategy:
    """Tests for the KeepBothStrategy"""
    
    def test_get_confidence_relationship(self, relationship_conflict):
        """Test confidence calculation for relationship conflicts"""
        strategy = KeepBothStrategy()
        confidence = strategy.get_confidence(relationship_conflict)
        
        # Should have medium-high confidence for relationship conflicts
        assert confidence >= 0.6
        
    def test_get_confidence_property(self, property_value_conflict):
        """Test confidence calculation for property conflicts"""
        strategy = KeepBothStrategy()
        confidence = strategy.get_confidence(property_value_conflict)
        
        # Should have zero confidence for property conflicts
        assert confidence == 0.0
        
    def test_get_resolution_option(self, relationship_conflict):
        """Test getting the correct resolution option"""
        strategy = KeepBothStrategy()
        option = strategy.get_resolution_option(relationship_conflict)
        
        assert option is not None
        assert option.id == "opt3"
        assert option.resolution_type == "keep_both_relationships"

class TestLLMBasedStrategy:
    """Tests for the LLMBasedStrategy"""
    
    def test_get_confidence_critical(self, critical_conflict):
        """Test confidence calculation for critical conflicts"""
        strategy = LLMBasedStrategy()
        confidence = strategy.get_confidence(critical_conflict)
        
        # Should have high confidence for critical conflicts
        assert confidence >= 0.7
        
    def test_get_confidence_minor(self, property_value_conflict):
        """Test confidence calculation for minor conflicts"""
        strategy = LLMBasedStrategy()
        confidence = strategy.get_confidence(property_value_conflict)
        
        # Should have lower confidence for minor conflicts
        assert confidence <= 0.5
        
    @pytest.mark.asyncio
    async def test_get_llm_recommendation(self, critical_conflict):
        """Test getting recommendations from LLM"""
        strategy = LLMBasedStrategy()
        
        # Mock the LLM response
        with patch('app.services.merge.strategy_selection.b.SelectBestResolution', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(resolution_id="opt1", confidence=0.9)
            
            resolution_id, confidence = await strategy.get_llm_recommendation(critical_conflict)
            
            assert resolution_id == "opt1"
            assert confidence == 0.9
            mock_llm.assert_called_once()

class TestStrategySelectionEngine:
    """Tests for the StrategySelectionEngine"""
    
    def test_init_with_default_strategies(self):
        """Test initialization with default strategies"""
        engine = StrategySelectionEngine()
        
        # Should have all default strategies
        assert "prefer_staging" in engine.strategies
        assert "prefer_production" in engine.strategies
        assert "merge_values" in engine.strategies
        assert "keep_both" in engine.strategies
        assert "llm_assisted" in engine.strategies
        
    def test_init_with_custom_strategies(self):
        """Test initialization with custom strategies"""
        config = {
            "custom_strategies": {
                "custom_rule": {
                    "type": "rule_based",
                    "description": "Custom rule-based strategy",
                    "base_confidence": 0.75,
                    "rules": {
                        "entity_types": {
                            "Person": {
                                "confidence": 0.8,
                                "resolution_type": "keep_staging_value"
                            }
                        },
                        "properties": {
                            "name": {
                                "confidence": 0.9,
                                "resolution_type": "keep_staging_value"
                            }
                        }
                    }
                }
            }
        }
        
        engine = StrategySelectionEngine(config)
        
        # Should have custom strategy
        assert "custom_rule" in engine.strategies
        
    @pytest.mark.asyncio
    async def test_select_strategy_property_value(self, property_value_conflict):
        """Test strategy selection for property value conflicts"""
        engine = StrategySelectionEngine()
        
        strategy_name, option, confidence, explanation = await engine.select_strategy(property_value_conflict)
        
        # Should select prefer_staging for this conflict
        assert strategy_name == "prefer_staging"
        assert option.id == "opt1"
        assert confidence > 0.7
        assert explanation is not None
        
    @pytest.mark.asyncio
    async def test_select_strategy_list_property(self, list_property_conflict):
        """Test strategy selection for list property conflicts"""
        engine = StrategySelectionEngine()
        
        strategy_name, option, confidence, explanation = await engine.select_strategy(list_property_conflict)
        
        # Should select merge_values for this conflict
        assert strategy_name == "merge_values"
        assert option.id == "opt3"
        assert confidence > 0.6
        assert explanation is not None
        
    @pytest.mark.asyncio
    async def test_select_strategy_relationship(self, relationship_conflict):
        """Test strategy selection for relationship conflicts"""
        engine = StrategySelectionEngine()
        
        strategy_name, option, confidence, explanation = await engine.select_strategy(relationship_conflict)
        
        # Should select either prefer_staging or keep_both
        assert strategy_name in ["prefer_staging", "keep_both"]
        assert option is not None
        assert confidence > 0.4
        assert explanation is not None
        
    @pytest.mark.asyncio
    async def test_select_strategy_with_llm(self, critical_conflict):
        """Test strategy selection with LLM for critical conflicts"""
        engine = StrategySelectionEngine({"always_use_llm": True})
        
        # Mock the LLM response
        with patch.object(LLMBasedStrategy, 'get_llm_recommendation', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("opt1", 0.9)
            
            strategy_name, option, confidence, explanation = await engine.select_strategy(critical_conflict)
            
            # Should use LLM for critical conflicts
            assert strategy_name == "llm_assisted"
            assert option.id == "opt1"
            assert confidence > 0.6
            assert explanation is not None
            mock_llm.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_select_strategy_with_config_override(self, property_value_conflict):
        """Test strategy selection with configuration overrides"""
        config = {
            "custom_strategies": {
                "custom_rule": {
                    "type": "rule_based",
                    "description": "Custom rule for Person entities",
                    "base_confidence": 0.95,
                    "rules": {
                        "entity_types": {
                            "Person": {
                                "confidence": 0.95,
                                "resolution_type": "keep_production_value"
                            }
                        }
                    }
                }
            }
        }
        
        engine = StrategySelectionEngine(config)
        
        strategy_name, option, confidence, explanation = await engine.select_strategy(property_value_conflict)
        
        # Should select custom rule due to high confidence
        assert strategy_name == "custom_rule"
        assert option.id == "opt2"  # production value option
        assert confidence > 0.9
        assert explanation is not None 