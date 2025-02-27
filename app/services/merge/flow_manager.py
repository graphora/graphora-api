"""Prefect flow deployment and invocation management"""
import asyncio
from prefect.client.orchestration import get_client

async def create_resolution_pipeline_deployment():
    """Create and deploy the resolution pipeline flow"""
    # Import here to avoid circular imports
    from app.services.merge.prefect_flows import resolution_pipeline_flow
    
    # Use the newer flow.deploy() method instead of Deployment.build_from_flow
    deployment = await resolution_pipeline_flow.deploy(
        name="resolution-pipeline-deployment",
        work_queue_name="merge-operations",
        tags=["merge", "resolution"],
    )
    
    return deployment.id

async def run_resolution_pipeline(merge_id: str) -> str:
    """Run the resolution pipeline for a specific merge"""
    # Import here to avoid circular imports
    from app.services.merge.prefect_flows import resolution_pipeline_flow
    
    async with get_client() as client:
        flow_run = await client.create_flow_run(
            flow=resolution_pipeline_flow,
            parameters={"merge_id": merge_id},
            name=f"Resolution-{merge_id}"
        )
        
        return flow_run.id

async def get_flow_run_status(flow_run_id: str):
    """Get the status of a flow run"""
    async with get_client() as client:
        flow_run = await client.read_flow_run(flow_run_id)
        return flow_run.state.name

async def cancel_flow_run(flow_run_id: str):
    """Cancel a running flow"""
    async with get_client() as client:
        await client.cancel_flow_run(flow_run_id)
        return True 