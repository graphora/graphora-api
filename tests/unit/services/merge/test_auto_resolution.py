"""Unit tests for auto resolution engine"""
import pytest
from unittest.mock import MagicMock, patch
import uuid
from datetime import datetime

from app.services.merge.auto_resolution import AutoResolutionEngine
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption, ResolutionStrategy

@pytest.fixture
def property_value_conflict():
    """Sample property value conflict for testing"""
    conflict_id = f"conflict-{uuid.uuid4().hex}"
    return Conflict(
        id=conflict_id,
        merge_id="test-merge-id",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MINOR,
        entity_id="s1",
        entity_type="Person",
        property_name="name",
        staging_value="Test Entity",
        production_value="test entity",
        description="Property 'name' has different values",
        resolution_options=[
            ResolutionOption(
                id=f"{conflict_id}_staging",
                description="Keep staging value: 'Test Entity'",
                resolution_type=ResolutionStrategy.KEEP_STAGING,
                resolution_data={"property_name": "name"},
                confidence=0.6
            ),
            ResolutionOption(
                id=f"{conflict_id}_prod",
                description="Keep production value: 'test entity'",
                resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                resolution_data={"property_name": "name"},
                confidence=0.4
            )
        ]
    )

@pytest.fixture
def missing_property_conflict():
    """Sample missing property conflict for testing"""
    conflict_id = f"conflict-{uuid.uuid4().hex}"
    return Conflict(
        id=conflict_id,
        merge_id="test-merge-id",
        conflict_type=ConflictType.PROPERTY_MISSING,
        severity=ConflictSeverity.MINOR,
        entity_id="s1",
        entity_type="Person",
        property_name="email",
        staging_value="user@example.com",
        production_value=None,
        description="Property 'email' exists in staging but not in production",
        context={
            "property_name": "email",
            "missing_in": "production",
            "entity_type": "Person"
        },
        resolution_options=[
            ResolutionOption(
                id=f"{conflict_id}_add",
                description="Add property to production",
                resolution_type=ResolutionStrategy.KEEP_STAGING,
                resolution_data={"property_name": "email"},
                confidence=0.8
            ),
            ResolutionOption(
                id=f"{conflict_id}_remove",
                description="Remove property from staging",
                resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                resolution_data={"property_name": "email"},
                confidence=0.2
            )
        ]
    )

@pytest.fixture
def numeric_property_conflict():
    """Sample numeric property conflict for testing"""
    conflict_id = f"conflict-{uuid.uuid4().hex}"
    return Conflict(
        id=conflict_id,
        merge_id="test-merge-id",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MINOR,
        entity_id="s1",
        entity_type="Person",
        property_name="age",
        staging_value=32,
        production_value=31,
        description="Property 'age' has different values",
        resolution_options=[
            ResolutionOption(
                id=f"{conflict_id}_staging",
                description="Keep staging value: 32",
                resolution_type=ResolutionStrategy.KEEP_STAGING,
                resolution_data={"property_name": "age"},
                confidence=0.5
            ),
            ResolutionOption(
                id=f"{conflict_id}_prod",
                description="Keep production value: 31",
                resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                resolution_data={"property_name": "age"},
                confidence=0.5
            )
        ]
    )

@pytest.fixture
def major_conflict():
    """Sample major conflict that shouldn't be auto-resolved"""
    conflict_id = f"conflict-{uuid.uuid4().hex}"
    return Conflict(
        id=conflict_id,
        merge_id="test-merge-id",
        conflict_type=ConflictType.RELATIONSHIP_TYPE,
        severity=ConflictSeverity.MAJOR,
        entity_id="s1",
        entity_type="Person",
        description="Different relationship types between the same entities",
        context={
            "staging_type": "WORKS_IN",
            "production_type": "BELONGS_TO",
            "entity_type": "Person"
        },
        resolution_options=[
            ResolutionOption(
                id=f"{conflict_id}_staging",
                description="Keep staging relationship type",
                resolution_type=ResolutionStrategy.KEEP_STAGING_REL,
                resolution_data={},
                confidence=0.5
            )
        ]
    )

class TestAutoResolutionEngine:
    @pytest.mark.asyncio
    async def test_case_difference_auto_resolution(self, property_value_conflict):
        """Test that case differences are auto-resolved"""
        engine = AutoResolutionEngine()
        resolution_id = await engine.resolve_conflict(property_value_conflict)
        
        assert resolution_id is not None
        assert resolution_id == f"{property_value_conflict.id}_staging"  # Should prefer staging value for case difference
        
    @pytest.mark.asyncio
    async def test_missing_property_auto_resolution(self, missing_property_conflict):
        """Test missing property auto-resolution"""
        engine = AutoResolutionEngine()
        resolution_id = await engine.resolve_conflict(missing_property_conflict)
        
        assert resolution_id is not None
        assert resolution_id == f"{missing_property_conflict.id}_add"  # Should add missing property to production
        
    @pytest.mark.asyncio
    async def test_numeric_difference_auto_resolution(self, numeric_property_conflict):
        """Test numeric difference auto-resolution"""
        engine = AutoResolutionEngine()
        resolution_id = await engine.resolve_conflict(numeric_property_conflict)
        
        assert resolution_id is not None
        assert resolution_id == f"{numeric_property_conflict.id}_staging"  # Should prefer larger value
        
    @pytest.mark.asyncio
    async def test_no_auto_resolution_for_major_conflicts(self, major_conflict):
        """Test that major conflicts aren't auto-resolved"""
        engine = AutoResolutionEngine()
        resolution_id = await engine.resolve_conflict(major_conflict)
        
        assert resolution_id is None  # Should not auto-resolve
        
    @pytest.mark.asyncio
    async def test_config_override(self, property_value_conflict):
        """Test config override for specific entity type"""
        config = {
            "Person": {
                "type": "prefer_production"
            }
        }
        
        engine = AutoResolutionEngine(config)
        resolution_id = await engine.resolve_conflict(property_value_conflict)
        
        assert resolution_id is not None
        assert resolution_id == f"{property_value_conflict.id}_prod"  # Should use config to prefer production
        
    @pytest.mark.asyncio
    async def test_property_specific_config(self, property_value_conflict):
        """Test property-specific config override"""
        config = {
            "Person.name": {
                "type": "prefer_production"
            }
        }
        
        engine = AutoResolutionEngine(config)
        resolution_id = await engine.resolve_conflict(property_value_conflict)
        
        assert resolution_id is not None
        assert resolution_id == f"{property_value_conflict.id}_prod"  # Should use property-specific config 