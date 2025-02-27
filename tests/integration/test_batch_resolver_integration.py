"""Integration tests for the batch resolver service"""
from unittest.mock import AsyncMock
import pytest
import os
from app.services.merge.service import MergeService
from app.services.merge.batch_resolver import BatchResolver
from app.services.storage.neo4j import Neo4jStorage
from app.schemas.conflicts import ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.graph import Node, Edge, GraphResponse
from tests.utils.mock_data_generator import MockDataGenerator
from app.config import settings

# Skip if not running integration tests
pytestmark = pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ, 
    reason="Integration tests skipped without INTEGRATION_TESTS environment variable"
)

@pytest.fixture
async def setup_test_data():
    """Set up test data in Neo4j and Redis for integration testing"""
    # Mock data generator
    generator = MockDataGenerator()
    
    # Create storage instances
    staging_storage = Neo4jStorage(
        uri=settings.STAGING_NEO4J_URI,
        username=settings.STAGING_NEO4J_USER,
        password=settings.STAGING_NEO4J_PASSWORD,
        database=settings.STAGING_NEO4J_DATABASE
    )
    
    prod_storage = Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )
    
    # Generate test data
    transform_id = "test_transform_batch"
    merge_id = "test_merge_batch"
    
    staging_graph, prod_graph, expected_conflicts = generator.generate_conflicting_graphs(
        transform_id=transform_id
    )
    
    # Store the graphs in Neo4j
    # (Simplified - would need real Neo4j insert code here)
    
    # Store conflicts in Redis
    merge_service = MergeService(staging_storage, prod_storage, progress_tracker=AsyncMock())
    await merge_service._store_conflicts(merge_id, expected_conflicts)
    
    # Return test context
    return {
        "transform_id": transform_id,
        "merge_id": merge_id,
        "expected_conflicts": expected_conflicts
    }

@pytest.mark.asyncio
async def test_real_batch_resolution(setup_test_data):
    """Test batch resolution with real storage backends"""
    # Skip this test if we're not in an environment with real Neo4j/Redis
    if "FULL_INTEGRATION_TESTS" not in os.environ:
        pytest.skip("Full integration test skipped without FULL_INTEGRATION_TESTS environment variable")
        
    # Create storage instances
    staging_storage = Neo4jStorage(
        uri=settings.STAGING_NEO4J_URI,
        username=settings.STAGING_NEO4J_USER,
        password=settings.STAGING_NEO4J_PASSWORD,
        database=settings.STAGING_NEO4J_DATABASE
    )
    
    prod_storage = Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )
    
    # Get test data
    merge_id = setup_test_data["merge_id"]
    expected_conflicts = setup_test_data["expected_conflicts"]
    
    # Create real merge service and batch resolver
    merge_service = MergeService(staging_storage, prod_storage, progress_tracker=AsyncMock())
    resolver = BatchResolver(merge_service)
    
    # Group conflicts
    groups = await resolver.group_similar_conflicts(
        merge_id=merge_id,
        grouping_strategy="type_and_entity"
    )
    
    # Verify groups match expectations
    assert len(groups) > 0
    
    # Find property conflict group
    property_group_key = next(
        (key for key in groups.keys() if key.startswith("property_value:")), 
        None
    )
    
    if property_group_key is None:
        pytest.skip("No property conflict group found in test data")
    
    # Create resolution option
    resolution_option = ResolutionOption(
        id="test_option",
        description="Keep staging value",
        resolution_type="keep_staging",
        resolution_data={},
        confidence=0.8
    )
    
    # Apply batch resolution
    result = await resolver.apply_batch_resolution(
        merge_id=merge_id,
        group_key=property_group_key,
        resolution_option=resolution_option
    )
    
    # Verify results
    assert result["status"] == "success"
    assert result["resolved_count"] > 0
    
    # Check conflicts were actually resolved in storage
    conflicts, _ = await merge_service.get_conflicts(
        merge_id=merge_id,
        resolved=True
    )
    
    # Should have at least as many resolved conflicts as our batch operation resolved
    assert len(conflicts) >= result["resolved_count"] 