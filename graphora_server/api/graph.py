from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict, List, Optional
from graphora_server.schemas.graph import GraphResponse
from graphora_server.schemas.graph_changes import SaveGraphRequest, SaveGraphResponse
from graphora_server.services.user_db_service import (
    UserDatabaseService,
    is_memory_storage_enabled,
)
from graphora_server.services.decision_log_service import (
    DecisionLogService,
    DecisionType,
)
from graphora_server.utils.logger import logger
import traceback
from graphora_server.auth import AuthContext, get_current_auth

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])


@router.get(
    "/{transform_id}",
    response_model=GraphResponse,
    description="Retrieve nodes by transform ID and their relationships",
)
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
                status_code=400, detail="Limit and skip must be non-negative"
            )

        if limit > 10000:
            raise HTTPException(status_code=400, detail="Maximum limit is 10000 nodes")

        # Check if user has staging DB configured or if memory storage is globally enabled
        from graphora_server.services.storage.factory import user_has_staging_db

        use_in_memory = is_memory_storage_enabled() or not await user_has_staging_db(
            auth.user_id
        )

        if use_in_memory:
            from graphora_server.services.storage.memory import InMemoryStorage

            storage = InMemoryStorage(user_id=auth.user_id)
            response = await storage.get_transformation_data(transform_id)

            # Apply pagination
            nodes = response.nodes[skip : skip + limit]
            response = GraphResponse(
                nodes=nodes,
                edges=response.edges,
                total_nodes=response.total_nodes,
                total_edges=response.total_edges,
                metadata=response.metadata,
            )

            logger.info(
                "Retrieved %s nodes and %s edges for user %s from in-memory storage",
                len(response.nodes),
                len(response.edges),
                auth.user_id,
            )
            return response

        # Get user's staging database (graph operations always use staging)
        graph_service = await UserDatabaseService.get_staging_graph_service(
            auth.user_id
        )

        # Get graph data
        response = graph_service.get_graph_by_transform_id(
            transform_id=transform_id, limit=limit, skip=skip
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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        logger.error(
            "Error retrieving graph data for user %s: %s", auth.user_id, str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Error retrieving graph data: {str(e)}"
        )
    finally:
        if graph_service:
            graph_service.close()


def _decision_to_dict(decision: Any) -> Dict[str, Any]:
    """Serialize a Decision dataclass into the JSON shape the
    /decisions response contract specifies. Enums are serialized
    to string values so the response survives JSON transport
    without requiring a Python-typed consumer.

    Mirror of the helper in graphora_server/mcp/server.py — kept
    duplicated rather than abstracted because the two surfaces
    have different consumers (REST API vs MCP tool) and the
    serialization choice could legitimately diverge later."""
    return {
        "id": decision.id,
        "transform_id": decision.transform_id,
        "target_id": decision.target_id,
        "target_kind": decision.target_kind.value,
        "decision_type": decision.decision_type.value,
        "reason": decision.reason,
        "evidence": decision.evidence,
        "alternatives": decision.alternatives,
        "created_at": decision.created_at,
    }


@router.get(
    "/{transform_id}/decisions",
    description=(
        "Decision Log entries for a transform. With ``node_id`` set, "
        "returns schema-level decisions plus the per-node decisions "
        "for that node, schema-first. Without ``node_id``, returns "
        "only schema-level decisions."
    ),
)
async def get_decisions_by_transform_id(
    transform_id: str,
    node_id: Optional[str] = None,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    """B0-explain (reviewer-flagged on commit 9ac9bb5): the Decision
    Log lives on the API side, not in MCP. MCP is documented and
    implemented as an HTTP client; reading decisions directly from
    a local DecisionLogService inside the MCP process would either
    fall into an empty in-memory store (no DATABASE_URL) or open a
    new direct DB dependency / secret surface (DATABASE_URL set).

    This endpoint owns the read so MCP stays a pure HTTP client.

    Schema-level decisions (target_kind=schema, decision_type=
    schema_inferred) are always included for context — they're
    transform-level prerequisites for any per-node "why is this
    here?" answer. Node decisions follow if ``node_id`` is supplied.

    Performance pin: schema decisions are fetched via
    ``for_decision_type`` (uses the (transform_id, decision_type)
    index from migration 14) so node-evidence lookups don't scale
    with the total decision count for the transform.

    Returns:
        decision_log (list): Schema decisions first, then node
            decisions if ``node_id`` is supplied. Each entry: ``{id,
            transform_id, target_id, target_kind, decision_type,
            reason, evidence, alternatives, created_at}``.
        alternatives (list): Aggregated candidate entities the
            pipeline considered for ``node_id`` across all merge
            decisions. Empty when ``node_id`` is None or the node
            had no merge events.
    """
    try:
        log = DecisionLogService()

        # Schema-level: narrow to decision_type for the index hit;
        # see for_decision_type docstring + reviewer's P3 finding.
        schema_decisions = await log.for_decision_type(
            transform_id, DecisionType.SCHEMA_INFERRED
        )

        node_decisions: List[Any] = []
        alternatives: List[Dict[str, Any]] = []
        if node_id:
            node_decisions = await log.for_target(transform_id, node_id)
            for d in node_decisions:
                alternatives.extend(d.alternatives)

        # Schema first (causation chain: schema is the prerequisite
        # for the node merges that followed). The Evidence tab
        # renders this as a top-down narrative; flattening to
        # walltime would mis-narrate causation for re-extractions
        # where node merges land before a schema decision in
        # walltime.
        decision_dicts = [
            _decision_to_dict(d) for d in schema_decisions + node_decisions
        ]

        return {
            "decision_log": decision_dicts,
            "alternatives": alternatives,
        }
    except Exception as e:
        traceback.print_exc()
        logger.error(
            "Error fetching decisions for transform %s, user %s: %s",
            transform_id,
            auth.user_id,
            str(e),
        )
        raise HTTPException(
            status_code=500, detail=f"Error fetching decisions: {str(e)}"
        )


@router.put(
    "/{transform_id}",
    response_model=SaveGraphResponse,
    description="Save graph changes in a single transaction",
)
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
        # Check if user has staging DB configured or if memory storage is globally enabled
        from graphora_server.services.storage.factory import user_has_staging_db

        use_in_memory = is_memory_storage_enabled() or not await user_has_staging_db(
            auth.user_id
        )

        if use_in_memory:
            # Use in-memory storage for saving changes
            from graphora_server.services.storage.memory import InMemoryStorage

            storage = InMemoryStorage(user_id=auth.user_id)
            result = await storage.save_graph_changes(transform_id, changes)

            logger.info(
                "Saved graph changes for user %s in in-memory storage",
                auth.user_id,
            )
            return result

        # Get user's staging database (graph operations always use staging)
        graph_service = await UserDatabaseService.get_staging_graph_service(
            auth.user_id
        )

        # Save changes
        result = graph_service.save_graph_changes(transform_id, changes)

        logger.info(
            "Saved graph changes for user %s in staging database",
            auth.user_id,
        )
        return result

    except ValueError as e:
        logger.error("Configuration error for user %s: %s", auth.user_id, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error saving graph changes for user %s: %s", auth.user_id, str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error saving changes: {str(e)}")
    finally:
        if graph_service:
            graph_service.close()
