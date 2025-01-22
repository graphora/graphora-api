from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from app.schemas.graph import GraphResponse
from app.services.graph_service import GraphService
from app.config import get_settings
from app.utils.logger import logger

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])

def get_staging_graph_service() -> GraphService:
    """Dependency to get graph service instance"""
    settings = get_settings()
    service = GraphService(
        uri=settings.STAGING_NEO4J_URI,
        user=settings.STAGING_NEO4J_USER,
        password=settings.STAGING_NEO4J_PASSWORD
    )
    try:
        return service
    finally:
        service.close()

@router.get("/{label}",
         response_model=GraphResponse,
         description="Retrieve nodes by label and their relationships")
def get_graph_by_label(
    label: str,
    limit: Optional[int] = 1000,
    skip: Optional[int] = 0,
    graph_service: GraphService = Depends(get_staging_graph_service)
) -> GraphResponse:
    """
    Retrieve nodes by label and their relationships
    
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

        # Get graph data
        response = graph_service.get_graph_by_label(
            label=f"Staging_{label}",
            limit=limit,
            skip=skip
        )
        
        if not response.nodes:
            logger.warning(f"No nodes found with label: {label}")
            
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving graph data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving graph data: {str(e)}"
        )
