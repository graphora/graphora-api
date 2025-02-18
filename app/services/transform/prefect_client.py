from typing import Optional, Dict, Any
import os
from prefect.client.schemas.objects import FlowRun
from prefect import get_client

from app.config import settings

def configure_prefect():
    """Configure Prefect client with settings"""
    if settings.PREFECT_API_URL:
        os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL
    if settings.PREFECT_API_KEY:
        os.environ["PREFECT_API_KEY"] = settings.PREFECT_API_KEY

async def get_flow_run(transform_id: str) -> Optional[FlowRun]:
    """Get flow run by transform ID"""
    try:
        async with get_client() as client:
            # Query for flow runs with matching tag
            flows = await client.read_flows(
                flow_filter={"name": {"any_": ["document-transformation"]}},
                flow_run_filter={"tags": {"all_": [transform_id]}}
            )
            
            if not flows or not flows[0].latest_flow_runs:
                return None
                
            return flows[0].latest_flow_runs[0]
            
    except Exception as e:
        print(f"Failed to get flow run: {str(e)}")
        return None

async def get_flow_run_data(
    transform_id: str
) -> Optional[Dict[str, Any]]:
    """Get flow run data by transform ID"""
    try:
        flow_run = await get_flow_run(transform_id)
        if not flow_run or not flow_run.state:
            return None
            
        return {
            'id': transform_id,
            'status': flow_run.state.type.value,
            'start_time': flow_run.start_time,
            'end_time': flow_run.end_time,
            'data': flow_run.state.data
        }
        
    except Exception as e:
        print(f"Failed to get flow run data: {str(e)}")
        return None
