"""LangGraph pipeline for conflict resolution"""
from typing import Dict, Any, TypedDict, Optional
import logging
import uuid

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

from app.schemas.conflicts import Conflict, ConflictType
from app.baml_client.async_client import b

logger = logging.getLogger(__name__)

class ConflictResolutionNode(BaseModel):
    """Node in conflict resolution workflow"""
    conflict_id: str
    merge_id: str
    status: str = "pending"
    resolution_id: Optional[str] = None
    error: Optional[str] = None

class ConflictState(TypedDict):
    conflict: Dict[str, Any]  # Serialized conflict
    merge_id: str
    ontology: Dict[str, Any]
    resolution: Optional[Dict[str, Any]]
    error: Optional[str]
    status: str

def build_resolution_pipeline() -> CompiledGraph:
    """Build a langgraph pipeline for conflict resolution"""
    graph = StateGraph(ConflictState)
    
    # Nodes in the graph
    graph.add_node("analyze_conflict", analyze_conflict)
    graph.add_node("generate_options", generate_options)
    graph.add_node("select_option", select_option)
    graph.add_node("apply_resolution", apply_resolution)
    graph.add_node("handle_error", handle_error)
    
    def route_analyze_conflict(state: ConflictState) -> str:
        return route_by_error(state, "generate_options")
    
    def route_generate_options(state: ConflictState) -> str:
        return route_by_error(state, "select_option")
    
    def route_select_option(state: ConflictState) -> str:
        return route_by_error(state, "apply_resolution")
    
    def route_apply_resolution(state: ConflictState) -> str:
        return route_by_error(state, END)
    
    # Define the edges
    graph.add_conditional_edges("analyze_conflict", route_analyze_conflict)
    graph.add_conditional_edges("generate_options", route_generate_options)
    graph.add_conditional_edges("select_option", route_select_option)
    graph.add_conditional_edges("apply_resolution", route_apply_resolution)
    
    graph.add_edge("handle_error", END)
    
    return graph.compile()

def route_by_error(state: ConflictState, next_step: str) -> str:
    """Check if state has an error"""
    if state.get("error") is not None:
        return "handle_error"
    else :
        return next_step

async def analyze_conflict(state: ConflictState) -> ConflictState:
    """Analyze the conflict using BAML"""
    try:
        conflict_data = state["conflict"]
        ontology_data = state["ontology"]
        
        # Create a Conflict object
        conflict = Conflict.model_validate(conflict_data)
        
        # Select the appropriate BAML function based on conflict type
        analysis_result = None
        
        if conflict.conflict_type == ConflictType.PROPERTY:
            analysis_result = await b.AnalyzePropertyConflictWithOntology(
                conflict=conflict.model_dump(),
                ontology=ontology_data
            )
        elif conflict.conflict_type == ConflictType.RELATIONSHIP:
            analysis_result = await b.AnalyzeRelationshipConflictWithOntology(
                conflict=conflict.model_dump(),
                ontology=ontology_data
            )
        elif conflict.conflict_type == ConflictType.ENTITY_MATCH:
            analysis_result = await b.AnalyzeEntityMatchConflictWithOntology(
                conflict=conflict.model_dump(),
                ontology=ontology_data
            )
        else:
            analysis_result = await b.AnalyzeGenericConflictWithOntology(
                conflict=conflict.model_dump(),
                ontology=ontology_data
            )
            
        # Update the conflict with analysis
        conflict.analysis = analysis_result.model_dump()
        
        # Update state
        state["conflict"] = conflict.model_dump()
        state["status"] = "analyzed"
        
        return state
        
    except Exception as e:
        logger.error(f"Error analyzing conflict: {str(e)}")
        state["error"] = f"Failed to analyze conflict: {str(e)}"
        return state

async def generate_options(state: ConflictState) -> ConflictState:
    """Generate resolution options for the conflict"""
    try:
        conflict_data = state["conflict"]
        ontology_data = state["ontology"]
        
        # Create a Conflict object
        conflict = Conflict.model_validate(conflict_data)
        
        # Select the appropriate BAML function based on conflict type
        options_result = None
        
        if conflict.conflict_type == ConflictType.PROPERTY:
            options_result = await b.GeneratePropertyResolutionOptions(
                conflict=conflict.model_dump(),
                analysis=conflict.analysis,
                ontology=ontology_data
            )
        elif conflict.conflict_type == ConflictType.RELATIONSHIP:
            options_result = await b.GenerateRelationshipResolutionOptions(
                conflict=conflict.model_dump(),
                analysis=conflict.analysis,
                ontology=ontology_data
            )
        elif conflict.conflict_type == ConflictType.ENTITY_MATCH:
            options_result = await b.GenerateEntityMatchResolutionOptions(
                conflict=conflict.model_dump(),
                analysis=conflict.analysis,
                ontology=ontology_data
            )
        else:
            options_result = await b.GenerateGenericResolutionOptions(
                conflict=conflict.model_dump(),
                analysis=conflict.analysis,
                ontology=ontology_data
            )
            
        # Update the conflict with resolution options
        conflict.resolution_options = options_result.options
        
        # Update state
        state["conflict"] = conflict.model_dump()
        state["status"] = "options_generated"
        
        return state
        
    except Exception as e:
        logger.error(f"Error generating resolution options: {str(e)}")
        state["error"] = f"Failed to generate resolution options: {str(e)}"
        return state

async def select_option(state: ConflictState) -> ConflictState:
    """Select the best resolution option"""
    try:
        conflict_data = state["conflict"]
        ontology_data = state["ontology"]
        
        # Create a Conflict object
        conflict = Conflict.model_validate(conflict_data)
        
        # Get resolution options
        options = {
            "options": conflict.resolution_options
        }
        
        # Use BAML to select the best option
        selected_resolution = await b.SelectBestResolution(
            conflict=conflict.model_dump(),
            options=options,
            ontology=ontology_data
        )
        
        # Store the selected resolution
        state["resolution"] = selected_resolution.model_dump()
        state["status"] = "option_selected"
        
        return state
        
    except Exception as e:
        logger.error(f"Error selecting resolution option: {str(e)}")
        state["error"] = f"Failed to select resolution option: {str(e)}"
        return state

async def apply_resolution(state: ConflictState) -> ConflictState:
    """Apply the selected resolution"""
    try:
        conflict_data = state["conflict"]
        resolution_data = state["resolution"]
        
        # Create a Conflict object
        conflict = Conflict.model_validate(conflict_data)
        
        # Find the selected resolution option
        selected_option_id = resolution_data["option_id"]
        selected_option = None
        
        for option in conflict.resolution_options:
            if option.id == selected_option_id:
                selected_option = option
                break
                
        if not selected_option:
            # Create a new option if not found
            for option in conflict.resolution_options:
                if option.description == resolution_data["description"]:
                    selected_option = option
                    break
                    
        if not selected_option:
            # Create a completely new option
            selected_option = {
                "id": str(uuid.uuid4()),
                "description": f"Auto-selected resolution: {resolution_data['resolution_type']}",
                "resolution_type": resolution_data["resolution_type"],
                "resolution_data": resolution_data["resolution_data"],
                "confidence": resolution_data["confidence"],
                "reasoning": "Selected by AI resolution pipeline"
            }
        
        # Mark conflict as resolved
        conflict.resolved = True
        conflict.resolution = selected_option
        
        # Update state
        state["conflict"] = conflict.model_dump()
        state["status"] = "resolved"
        
        return state
        
    except Exception as e:
        logger.error(f"Error applying resolution: {str(e)}")
        state["error"] = f"Failed to apply resolution: {str(e)}"
        return state

async def handle_error(state: ConflictState) -> ConflictState:
    """Handle errors in the resolution pipeline"""
    # Log the error
    logger.error(f"Error in conflict resolution pipeline: {state.get('error')}")
    
    # Update status
    state["status"] = "error"
    
    return state
