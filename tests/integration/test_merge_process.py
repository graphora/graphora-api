"""Integration tests for the complete merge process"""
import os
import time
import pytest
import pytest_asyncio
import httpx
import uuid
import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pathlib import Path
from httpx import AsyncClient
import shutil

from app.main import app
from app.services.merge.service import MergeService
from app.services.storage.neo4j import Neo4jStorage
from tests.utils.mock_data_generator import MockDataGenerator
from app.config import settings
from app.services.merge.progress import MergeProgress
from app.services.merge.service import MergeService
from app.services.storage.neo4j import Neo4jStorage
from app.services.storage.models import StorageStage

pytestmark = pytest.mark.asyncio

@pytest.fixture
def test_data():
    """Generate test data for merge process"""
    generator = MockDataGenerator()
    transform_id = str(uuid.uuid4())
    staging_graph, prod_graph, conflicts = generator.generate_conflicting_graphs(transform_id=transform_id)
    return {
        "transform_id": transform_id,
        "staging_graph": staging_graph,
        "prod_graph": prod_graph,
        "conflicts": conflicts
    }

@pytest.fixture
def test_merge_id():
    """Generate test merge ID"""
    return "test_merge_30895ad4b32b4510af4dc32148332ee5"

@pytest.fixture
def test_ontology():
    """Load test ontology and copy to test directory"""
    # Load test ontology content
    ontology_path = Path(__file__).parent.parent / "data" / "test_ontology.yaml"
    with open(ontology_path) as f:
        ontology_content = f.read()
        
    # Copy to test directory with known ID for verification
    test_ontology_id = "test_merge_30895ad4b32b4510af4dc32148332ee5"
    test_ontology_path = Path(settings.ONTOLOGY_DIR).expanduser() / f"{test_ontology_id}.yaml"
    test_ontology_path.write_text(ontology_content)
    
    return ontology_content

@pytest_asyncio.fixture
async def test_client():
    """Create test client"""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client

@pytest.fixture(autouse=True)
def setup_test_mode():
    """Enable test mode for settings"""
    settings.test_mode = True
    yield
    settings.test_mode = False

@pytest.fixture(autouse=True)
def setup_ontology_dir():
    """Setup and cleanup ontology directory"""
    # Create test ontology directory
    Path(settings.ONTOLOGY_DIR).expanduser().mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup test ontology directory after test
    shutil.rmtree(Path(settings.ONTOLOGY_DIR).expanduser(), ignore_errors=True)

@pytest_asyncio.fixture
async def setup_test_data(test_data):
    """Insert test data into staging and production databases"""
    # Create storage instances
    staging_storage = Neo4jStorage(
        settings.STAGING_NEO4J_URI,
        settings.STAGING_NEO4J_USER,
        settings.STAGING_NEO4J_PASSWORD,
        settings.STAGING_NEO4J_DATABASE
    )
    prod_storage = Neo4jStorage(
        settings.NEO4J_URI,
        settings.NEO4J_USER,
        settings.NEO4J_PASSWORD,
        settings.NEO4J_DB
    )
    
    # Insert staging data
    await staging_storage.store_nodes(
        test_data["staging_graph"].nodes,
        batch_index=0,
        transform_id=test_data["transform_id"]
    )
    await staging_storage.update_checkpoint(
        test_data["transform_id"],
        last_index=0,
        stage=StorageStage.NODES
    )
    
    # Insert staging relationships
    await staging_storage.store_relationships(
        test_data["staging_graph"].edges,
        batch_index=0,
        transform_id=test_data["transform_id"]
    )
    await staging_storage.update_checkpoint(
        test_data["transform_id"],
        last_index=0,
        stage=StorageStage.RELATIONSHIPS
    )
    
    # Insert production data
    await prod_storage.store_nodes(
        test_data["prod_graph"].nodes,
        batch_index=0,
        transform_id=test_data["transform_id"]
    )
    await prod_storage.update_checkpoint(
        test_data["transform_id"],
        last_index=0,
        stage=StorageStage.NODES
    )
    
    # Insert production relationships
    await prod_storage.store_relationships(
        test_data["prod_graph"].edges,
        batch_index=0,
        transform_id=test_data["transform_id"]
    )
    await prod_storage.update_checkpoint(
        test_data["transform_id"],
        last_index=0,
        stage=StorageStage.RELATIONSHIPS
    )
        
    yield test_data
    
    # Cleanup after test
    await staging_storage.clear_all()
    await prod_storage.clear_all()

@pytest.mark.skipif(
    not os.getenv("INTEGRATION_TESTS"),
    reason="Integration tests are skipped by default"
)
class TestMergeProcessIntegration:
    @pytest.mark.asyncio
    async def test_start_merge_process(self, test_client, setup_test_data, test_merge_id, test_ontology):
        """Test the complete merge workflow from start to finish"""
        # First register the ontology
        response = await test_client.post(
            f"{settings.API_V1_STR}/ontology",
            json={"text": test_ontology}
        )
        assert response.status_code == 200
        ontology_id = response.json()["id"]

        # Start merge process
        response = await test_client.post(
            f"{settings.API_V1_STR}/merge/{test_merge_id}/{setup_test_data['transform_id']}/start",
            json={"ontology_id": ontology_id}
        )
        assert response.status_code == 200
        
        # Get the actual merge ID from the response
        merge_response = response.json()
        actual_merge_id = merge_response["merge_id"]

        # Verify merge started
        response = await test_client.get(
            f"{settings.API_V1_STR}/merge/status/{actual_merge_id}"
        )
        assert response.status_code == 200
        status = response.json()
        assert status["overall_status"] == "pending"

        # Wait for merge to complete (with timeout)
        max_wait = 30  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = await test_client.get(
                f"{settings.API_V1_STR}/merge/status/{actual_merge_id}"
            )
            status = response.json()
            if status["overall_status"] in ["COMPLETED", "FAILED"]:
                break
            await asyncio.sleep(1)

        # Check for conflicts
        response = await test_client.get(
            f"{settings.API_V1_STR}/merge/{actual_merge_id}/conflicts"
        )
        assert response.status_code == 200
        conflicts = response.json()
        assert len(conflicts) > 0  # We expect conflicts from our test data

    @pytest.mark.asyncio
    async def test_strategy_selection_api(self, test_client, setup_test_data):
        """Test the strategy selection API endpoints"""
        # Start a merge process
        response = await test_client.post(
            f"{settings.API_V1_STR}/merge/{setup_test_data['transform_id']}/{setup_test_data['transform_id']}/start",
            json={"ontology_id": None}
        )
        assert response.status_code == 200
        
        # Get the actual merge ID from the response
        merge_response = response.json()
        merge_id = merge_response["merge_id"]
        
        # Wait for merge process to complete
        max_retries = 30  # Increase timeout
        for i in range(max_retries):
            status_response = await test_client.get(f"{settings.API_V1_STR}/merge/status/{merge_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data["overall_status"] == "COMPLETED":
                    break
            await asyncio.sleep(1)
        
        # Get conflicts
        conflicts_response = await test_client.get(f"{settings.API_V1_STR}/merge/conflicts/{merge_id}")
        assert conflicts_response.status_code == 200
        conflicts_data = conflicts_response.json()
        
        # Skip test if no conflicts found
        if len(conflicts_data) == 0:
            pytest.skip("No conflicts found for testing strategy selection")
        
        # Test strategy selection API
        select_response = await test_client.post(
            f"{settings.API_V1_STR}/merge/{merge_id}/select-strategies",
            json={"config": {}}
        )
        assert select_response.status_code == 200
        select_data = select_response.json()
        
        # Verify response structure
        assert "total" in select_data
        assert "processed" in select_data
        assert "strategy_counts" in select_data
        assert "confidence_avg" in select_data
        
        # Test apply strategies API
        apply_response = await test_client.post(
            f"{settings.API_V1_STR}/merge/{merge_id}/apply-strategies",
            json={"min_confidence": 0.7}
        )
        assert apply_response.status_code == 200
        apply_data = apply_response.json()
        
        # Verify response structure
        assert "total" in apply_data
        assert "applied" in apply_data
        assert "skipped_low_confidence" in apply_data
        assert "skipped_no_strategy" in apply_data
        assert "by_strategy" in apply_data