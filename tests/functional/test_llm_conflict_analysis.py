"""Functional tests for LLM-assisted conflict analysis API."""
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
import json

from app.services.merge.llm_analyzer import LLMConflictAnalyzer
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption


@pytest.mark.asyncio
async def test_analyze_conflicts_api(client: AsyncClient):
    """Test the analyze conflicts API endpoint."""
    # First create a merge
    merge_response = await client.post(
        "/api/v1/merge",
        json={
            "source_id": "source123",
            "target_id": "target456",
            "name": "Test Merge for LLM Analysis"
        }
    )
    assert merge_response.status_code == 200
    merge_data = merge_response.json()
    merge_id = merge_data["id"]
    
    # Mock the LLMConflictAnalyzer.analyze_conflict method
    with patch.object(LLMConflictAnalyzer, 'analyze_conflict', new_callable=AsyncMock) as mock_analyzer:
        # Set up the mock to return sample resolution options
        mock_analyzer.return_value = [
            ResolutionOption(
                id="option1",
                description="Use value: 42",
                resolution_type="KEEP_STAGING",
                resolution_data={"property_name": "age", "value": "42"},
                confidence=0.8,
                reasoning="This is the correct answer based on analysis.",
                requires_review=False,
                auto_resolvable=True
            ),
            ResolutionOption(
                id="option2",
                description="Use value: 24",
                resolution_type="KEEP_PRODUCTION",
                resolution_data={"property_name": "age", "value": "24"},
                confidence=0.2,
                reasoning="Alternative value with lower confidence.",
                requires_review=True,
                auto_resolvable=False
            )
        ]
        
        # Call the analyze conflicts endpoint
        response = await client.post(
            f"/api/v1/merge/{merge_id}/conflicts/analyze",
            json={"conflict_ids": ["conflict123"]}
        )
        
        # Check response
        assert response.status_code == 200
        result = response.json()
        assert "total_conflicts" in result
        assert "analyzed" in result


@pytest.mark.asyncio
async def test_analyze_conflicts_with_no_ids(client: AsyncClient):
    """Test analyzing all conflicts by not providing conflict_ids."""
    # First create a merge
    merge_response = await client.post(
        "/api/v1/merge",
        json={
            "source_id": "source789",
            "target_id": "target987",
            "name": "Test Merge for All Conflicts LLM Analysis"
        }
    )
    assert merge_response.status_code == 200
    merge_data = merge_response.json()
    merge_id = merge_data["id"]
    
    # Mock both the get_conflicts method and the analyze_conflict method
    with patch('app.services.merge.service.MergeService.get_conflicts', new_callable=AsyncMock) as mock_get, \
         patch.object(LLMConflictAnalyzer, 'analyze_conflict', new_callable=AsyncMock) as mock_analyzer:
        
        # Mock get_conflicts to return sample conflicts
        test_conflict = Conflict(
            id="conflict1",
            conflict_type=ConflictType.PROPERTY,
            severity=ConflictSeverity.MAJOR,
            entity_id="entity1",
            property_name="age",
            staging_value="25",
            production_value="30",
            entity_type="Person",
            resolved=False,
            merge_id="",
            description="",
            resolution=None,
            analysis=None
        )
        mock_get.return_value = ([test_conflict], 1)  # List of conflicts and total count
        
        # Mock analyze_conflict to return sample options
        mock_analyzer.return_value = [
            ResolutionOption(
                id="option1",
                description="Use value: 30",
                resolution_type="KEEP_PRODUCTION",
                resolution_data={"property_name": "age", "value": "30"},
                confidence=0.8,
                reasoning="Target value is more recent.",
                requires_review=False,
                auto_resolvable=True
            )
        ]
        
        # Call the analyze conflicts endpoint without providing conflict_ids
        response = await client.post(
            f"/api/v1/merge/{merge_id}/conflicts/analyze"
        )
        
        # Check response
        assert response.status_code == 200
        result = response.json()
        assert result["total_conflicts"] == 1
        assert result["analyzed"] == 1
        
        # Verify the get_conflicts method was called with the correct parameters
        mock_get.assert_called_once_with(merge_id=merge_id, resolved=False)
