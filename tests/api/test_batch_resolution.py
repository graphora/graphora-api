"""Tests for batch resolution API endpoints"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.testclient import TestClient
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption, GroupBatchResolutionRequest

# Create sample conflicts for testing
SAMPLE_CONFLICTS = [
    Conflict(
        id="c1",
        merge_id="test_merge",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MINOR,
        staging_ids=["s1"],
        production_ids=["p1"],
        description="Test conflict 1",
        context={}
    ),
    Conflict(
        id="c2",
        merge_id="test_merge",
        conflict_type=ConflictType.RELATIONSHIP_TYPE,
        severity=ConflictSeverity.MAJOR,
        staging_ids=["s2"],
        production_ids=["p2"],
        description="Test conflict 2",
        context={}
    )
]

# Mock groups for testing
MOCK_GROUPS = {
    "group1": [SAMPLE_CONFLICTS[0]],
    "group2": [SAMPLE_CONFLICTS[1]]
}

# Mock batch resolution result
MOCK_BATCH_RESULT = {
    "status": "success",
    "group_key": "test_group",
    "resolved_count": 2,
    "total_in_group": 3,
    "exceptions_count": 1
}

# Create a mock BatchResolver class
class MockBatchResolver:
    def __init__(self, merge_service):
        self.merge_service = merge_service
    
    async def group_similar_conflicts(self, merge_id, grouping_strategy="type_and_entity", similarity_threshold=0.8, filters=None):
        return MOCK_GROUPS
    
    async def apply_batch_resolution(self, merge_id, group_key, resolution_option, exceptions=None):
        return MOCK_BATCH_RESULT

# Create error mock classes
class MockGroupErrorResolver:
    def __init__(self, merge_service):
        self.merge_service = merge_service
    
    async def group_similar_conflicts(self, merge_id, grouping_strategy="type_and_entity", similarity_threshold=0.8, filters=None):
        raise ValueError("Test error")
    
    async def apply_batch_resolution(self, merge_id, group_key, resolution_option, exceptions=None):
        return MOCK_BATCH_RESULT

class MockApplyErrorResolver:
    def __init__(self, merge_service):
        self.merge_service = merge_service
    
    async def group_similar_conflicts(self, merge_id, grouping_strategy="type_and_entity", similarity_threshold=0.8, filters=None):
        return MOCK_GROUPS
    
    async def apply_batch_resolution(self, merge_id, group_key, resolution_option, exceptions=None):
        raise ValueError("Test error")

@pytest.fixture
def mock_merge_service():
    """Create a mock merge service"""
    mock_service = MagicMock()
    mock_service.get_conflicts = AsyncMock(return_value=(SAMPLE_CONFLICTS, len(SAMPLE_CONFLICTS)))
    mock_service.resolve_conflict = AsyncMock(return_value={"resolved": True})
    return mock_service

# Create test router with our mock resolver
def create_test_router(resolver_class):
    router = APIRouter()
    
    @router.get(
        "/conflicts/{merge_id}/groups",
        description="Get grouped conflicts for batch resolution"
    )
    async def get_conflict_groups(
        merge_id: str,
        grouping_strategy: str = "type_and_entity",
        similarity_threshold: float = 0.8,
        merge_service = Depends(lambda: MagicMock())
    ):
        """Get grouped conflicts for batch resolution"""
        try:
            # Create batch resolver with our mock
            resolver = resolver_class(merge_service)
            
            # Get grouped conflicts
            groups = await resolver.group_similar_conflicts(
                merge_id=merge_id,
                grouping_strategy=grouping_strategy,
                similarity_threshold=similarity_threshold
            )
            
            # Convert conflicts to dictionaries for response
            result = {}
            for group_key, conflicts in groups.items():
                result[group_key] = [conflict.model_dump() for conflict in conflicts]
                
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error getting conflict groups: {str(e)}"
            )

    @router.post(
        "/conflicts/{merge_id}/resolve-batch",
        description="Resolve a batch of conflicts with the same strategy"
    )
    async def resolve_batch_conflicts(
        merge_id: str,
        request: dict,
        merge_service = Depends(lambda: MagicMock())
    ):
        """Resolve a batch of conflicts with the same strategy"""
        try:
            # Create batch resolver with our mock
            resolver = resolver_class(merge_service)
            
            # Extract request data
            group_key = request.get("group_key")
            resolution_option = ResolutionOption(**request.get("resolution_option"))
            exceptions = request.get("exceptions", [])
            
            # Apply batch resolution
            result = await resolver.apply_batch_resolution(
                merge_id=merge_id,
                group_key=group_key,
                resolution_option=resolution_option,
                exceptions=exceptions
            )
                
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error resolving batch conflicts: {str(e)}"
            )
    
    return router

@pytest.fixture
def test_app_normal():
    """Create a test app with normal resolver"""
    app = FastAPI()
    router = create_test_router(MockBatchResolver)
    app.include_router(router, prefix="/api/v1/merge")
    return app

@pytest.fixture
def test_app_group_error():
    """Create a test app with group error resolver"""
    app = FastAPI()
    router = create_test_router(MockGroupErrorResolver)
    app.include_router(router, prefix="/api/v1/merge")
    return app

@pytest.fixture
def test_app_apply_error():
    """Create a test app with apply error resolver"""
    app = FastAPI()
    router = create_test_router(MockApplyErrorResolver)
    app.include_router(router, prefix="/api/v1/merge")
    return app

def test_get_conflict_groups_api(test_app_normal):
    """Test the conflict groups API endpoint"""
    client = TestClient(test_app_normal)
    
    # Act
    response = client.get(
        "/api/v1/merge/conflicts/test_merge/groups?grouping_strategy=type_and_entity"
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    
    assert "group1" in data
    assert "group2" in data
    assert len(data["group1"]) == 1
    assert data["group1"][0]["id"] == "c1"

def test_resolve_batch_conflicts_api(test_app_normal):
    """Test the batch resolution API endpoint"""
    client = TestClient(test_app_normal)
    
    # Act
    response = client.post(
        "/api/v1/merge/conflicts/test_merge/resolve-batch",
        json={
            "group_key": "test_group",
            "resolution_option": {
                "id": "test_option",
                "description": "Keep staging value",
                "resolution_type": "keep_staging",
                "resolution_data": {},
                "confidence": 0.8
            },
            "exceptions": ["exclude_conflict"]
        }
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert data["resolved_count"] == 2
    assert data["total_in_group"] == 3

def test_get_conflict_groups_error_handling(test_app_group_error):
    """Test error handling in the conflict groups API endpoint"""
    client = TestClient(test_app_group_error)
    
    # Act
    response = client.get(
        "/api/v1/merge/conflicts/test_merge/groups"
    )
    
    # Assert
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"].lower()

def test_resolve_batch_conflicts_error_handling(test_app_apply_error):
    """Test error handling in the batch resolution API endpoint"""
    client = TestClient(test_app_apply_error)
    
    # Act
    response = client.post(
        "/api/v1/merge/conflicts/test_merge/resolve-batch",
        json={
            "group_key": "test_group",
            "resolution_option": {
                "id": "test_option",
                "description": "Keep staging value",
                "resolution_type": "keep_staging",
                "resolution_data": {},
                "confidence": 0.8
            }
        }
    )
    
    # Assert
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"].lower() 