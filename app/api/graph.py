from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from app.schemas.graph import GraphResponse
from app.schemas.graph_changes import SaveGraphRequest, SaveGraphResponse
from app.services.graph_service import GraphService
from app.services.user_db_service import UserDatabaseService
from app.config import settings
from app.utils.logger import logger
import traceback
from app.auth import AuthContext, get_current_auth

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])

@router.get("/{transform_id}",
         response_model=GraphResponse,
         description="Retrieve nodes by transform ID and their relationships")
async def get_graph_by_transform_id(
    transform_id: str,
    limit: Optional[int] = 1000,
    skip: Optional[int] = 0,
    auth: AuthContext = Depends(get_current_auth),
) -> GraphResponse:
    """
    Retrieve nodes by transform ID and their relationships from user's staging database
    
    Parameters:
    - transform_id: Transform ID to query
    - user_id: User's ID (from header)
    - limit: Maximum number of nodes to return (default: 1000)
    - skip: Number of nodes to skip for pagination (default: 0)
    
    Returns:
    - GraphResponse containing:
        - nodes: Array of nodes with properties
        - edges: Array of relationships between nodes
        - total_nodes: Total count of nodes with this label
        - total_edges: Total count of relationships
    """
    graph_service = None
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

        # Get user's staging database (graph operations always use staging)
        graph_service = await UserDatabaseService.get_staging_graph_service(auth.user_id)
        
        # Get graph data
        response = graph_service.get_graph_by_transform_id(
            transform_id=transform_id,
            limit=limit,
            skip=skip
        )
        
        if not response.nodes:
            logger.warning(
                "No nodes found with transform_id %s for user %s",
                transform_id,
                auth.user_id,
            )

        logger.info(
            "Retrieved %s nodes and %s edges for user %s from staging database",
            len(response.nodes),
            len(response.edges),
            auth.user_id,
        )
        return response
        
    except ValueError as e:
        logger.error("Configuration error for user %s: %s", auth.user_id, str(e))
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        traceback.print_exc()
        logger.error("Error retrieving graph data for user %s: %s", auth.user_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving graph data: {str(e)}"
        )
    finally:
        if graph_service:
            graph_service.close()

@router.put("/{transform_id}",
         response_model=SaveGraphResponse,
         description="Save graph changes in a single transaction")
async def save_graph_changes(
    transform_id: str,
    changes: SaveGraphRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> SaveGraphResponse:
    """
    Save bulk modifications to the user's staging graph database
    
    Parameters:
    - transform_id: Transformation ID
    - changes: Batch of modifications to apply
    - user_id: User's ID (from header)
    
    Returns:
    - Updated graph data
    - New version
    - Warning/info messages
    """
    graph_service = None
    try:
        
        # Get user's staging database (graph operations always use staging)
        graph_service = await UserDatabaseService.get_staging_graph_service(auth.user_id)
        
        # Save changes
        result = graph_service.save_graph_changes(transform_id, changes)
        
        logger.info(
            "Saved graph changes for user %s in staging database",
            auth.user_id,
        )
        return result
        
    except ValueError as e:
        logger.error("Configuration error for user %s: %s", auth.user_id, str(e))
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error saving graph changes for user %s: %s", auth.user_id, str(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error saving changes: {str(e)}"
        )
    finally:
        if graph_service:
            graph_service.close()
