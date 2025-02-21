from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from app.schemas.graph import GraphResponse
from app.schemas.graph_changes import SaveGraphRequest, SaveGraphResponse
from app.services.graph_service import GraphService
from app.config import settings
from app.utils.logger import logger
from app.utils.mock import transform_graph
import traceback

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])

def get_staging_graph_service() -> GraphService:
    """Dependency to get graph service instance"""
    service = GraphService(
        uri=settings.STAGING_NEO4J_URI,
        user=settings.STAGING_NEO4J_USER,
        password=settings.STAGING_NEO4J_PASSWORD
    )
    try:
        return service
    finally:
        service.close()

@router.get("/{transform_id}",
         response_model=GraphResponse,
         description="Retrieve nodes by transform ID and their relationships")
def get_graph_by_transform_id(
    transform_id: str,
    limit: Optional[int] = 1000,
    skip: Optional[int] = 0,
    graph_service: GraphService = Depends(get_staging_graph_service)
) -> GraphResponse:
    """
    Retrieve nodes by transform ID and their relationships
    
    Parameters:
    - label: Node label to query
    - limit: Maximum number of nodes to return (default: 1000)
    - skip: Number of nodes to skip for pagination (default: 0)
    
    Returns:
    - GraphResponse containing:
        - nodes: Array of nodes with properties
        - edges: Array of relationships between nodes
        - total_nodes: Total count of nodes with this label
        - total_edges: Total count of relationships
    """
    try:
        # Validate inputs
        if limit < 0 or skip < 0:
            raise HTTPException(
                status_code=400,
                detail="Limit and skip must be non-negative"
            )
        
        if limit > 10000:
            raise HTTPException(
                status_code=400,
                detail="Maximum limit is 10000 nodes"
            )
        if settings.MOCK_MODE:
            logger.info("Mock mode enabled, skipping document processing")
            return transform_graph

        # Get graph data
        response = graph_service.get_graph_by_transform_id(
            transform_id=transform_id,
            limit=limit,
            skip=skip
        )
        
        if not response.nodes:
            logger.warning(f"No nodes found with transform_id: {transform_id}")
            
        return response
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error retrieving graph data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving graph data: {str(e)}"
        )

@router.put("/{transform_id}",
         response_model=SaveGraphResponse,
         description="Save graph changes in a single transaction")
def save_graph_changes(
    transform_id: str,
    changes: SaveGraphRequest,
    graph_service: GraphService = Depends(get_staging_graph_service)
) -> SaveGraphResponse:
    """
    Save bulk modifications to the graph
    
    Parameters:
    - transform_id: Transformation ID
    - changes: Batch of modifications to apply
    
    Returns:
    - Updated graph data
    - New version
    - Warning/info messages
    """
    try:
        if settings.MOCK_MODE:
            logger.info("Mock mode enabled, skipping document processing")
            return SaveGraphResponse(
                data=transform_graph,
                messages=None
            )
        return graph_service.save_graph_changes(transform_id, changes)
    except ValueError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error saving graph changes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error saving changes: {str(e)}"
        )
