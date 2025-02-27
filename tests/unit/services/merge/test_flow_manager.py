"""Tests for Prefect flow deployment and invocation management"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prefect.testing.utilities import prefect_test_harness
from app.services.merge.flow_manager import (
    create_resolution_pipeline_deployment,
    run_resolution_pipeline
)

@pytest.fixture(autouse=True)
def prefect_test_fixture():
    with prefect_test_harness():
        yield

@pytest.mark.asyncio
async def test_create_resolution_pipeline_deployment():
    """Test creating a deployment for the resolution pipeline"""
    with patch("app.services.merge.prefect_flows.resolution_pipeline_flow") as mock_flow:
        # Setup mock deployment
        mock_deployment = MagicMock()
        mock_flow.deploy = AsyncMock(return_value=mock_deployment)
        mock_deployment.id = "test-deployment-id"
        
        # Call the function
        deployment_id = await create_resolution_pipeline_deployment()
        
        # Verify deployment was created with correct parameters
        assert deployment_id == "test-deployment-id"
        mock_flow.deploy.assert_called_once_with(
            name="resolution-pipeline-deployment",
            work_queue_name="merge-operations",
            tags=["merge", "resolution"],
        )

@pytest.mark.asyncio
async def test_run_resolution_pipeline():
    """Test running the resolution pipeline"""
    with patch("app.services.merge.flow_manager.get_client") as mock_get_client, \
         patch("app.services.merge.prefect_flows.resolution_pipeline_flow") as mock_flow:
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_get_client.return_value.__aenter__.return_value = mock_client
        
        # Setup mock flow run
        mock_flow_run = MagicMock()
        mock_flow_run.id = "test-flow-run-id"
        mock_client.create_flow_run = AsyncMock(return_value=mock_flow_run)
        
        # Call the function
        merge_id = "test-merge-id"
        flow_run_id = await run_resolution_pipeline(merge_id)
        
        # Verify flow run was created with correct parameters
        assert flow_run_id == "test-flow-run-id"
        mock_client.create_flow_run.assert_called_once_with(
            flow=mock_flow,
            parameters={"merge_id": merge_id},
            name=f"Resolution-{merge_id}"
        ) 