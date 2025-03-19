import re
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Type
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_core.runnables import RunnableConfig
import PyPDF2
import uuid
from google.auth import default
import google.auth.transport.requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from app.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    KnowledgeGraph,
    DocumentKnowledgeGraph,
    ExtractionMetrics
)
from pydantic import BaseModel, Field
from app.services.transform.agents.ontology import OntologyParser
project_id = "graphit-sandbox"
location = "us-central1"
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(google.auth.transport.requests.Request())

gemini_model = ChatOpenAI(
    base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
    api_key=credentials.token,
    model='google/gemini-2.0-flash-lite-001'
)


# Custom JSON encoder (unchanged)
def custom_json_encoder(obj):
    if isinstance(obj, (BaseNode, RelationshipInstance, NodeProvenance, ExtractionMetrics, DocumentKnowledgeGraph)):
        return obj.model_dump()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# State class (unchanged)
class GraphState(BaseModel):
    pdf_paths: List[str] = Field(default_factory=list)
    ontology: Dict[str, Any] = Field(default_factory=dict)
    ontology_yaml: str = Field(default="")
    entities_only_model: Optional[Type[BaseModel]] = None
    relationships_only_model: Optional[Type[BaseModel]] = None
    graph: DocumentKnowledgeGraph = Field(default_factory=DocumentKnowledgeGraph)
    context: str = Field(default="")
    decision_log: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    messages: List[Dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

# Helper to extract JSON (unchanged)
def extract_json_from_response(response_text: str) -> Optional[Dict]:
    json_pattern = r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})'
    match = re.search(json_pattern, response_text)
    if match:
        json_str = match.group(1) or match.group(2)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {json_str}, Error: {str(e)}")
            return None
    logger.error(f"No valid JSON found in response: {response_text}")
    return None

# Tools for Graph Construction
def extract_initial_graph(state: GraphState) -> Dict:
    """Extract initial graph from PDFs with detailed properties."""
    result = extract_structured_data(state.pdf_paths[0], ontology=state.ontology, 
                                     ontology_yaml=state.ontology_yaml, 
                                     entities_only_model=state.entities_only_model, 
                                     relationships_only_model=state.relationships_only_model)
    state.decision_log.append("Initial graph extracted from PDFs")
    return result

def compare_nodes(nodes: List[BaseNode], ontology: Dict[str, Any], context: str) -> Dict[str, List[Dict]]:
    """Compare all nodes of the same type in one shot using Gemini (unchanged)."""
    if not nodes or len(nodes) <= 1:
        return {"clusters": [{"nodes": [node.model_dump_json() for node in nodes], "confidence": 1.0, "reason": "Single or no nodes"}]}

    node_type = nodes[0].type
    if not all(node.type == node_type for node in nodes):
        raise ValueError("All nodes must be of the same type for one-shot comparison")

    print(f"Comparing {len(nodes)} nodes of type {node_type} in one shot")
    prompt = (
        f"Given the ontology and context, group these nodes of type '{node_type}' into clusters where nodes in each cluster refer to the same entity.\n"
        f"Ontology: {json.dumps(ontology)}\n"
        f"Nodes: {json.dumps([node.model_dump_json() for node in nodes], indent=2)}\n"
        f"Context: {context[:10000]}\n"
        f'Return only a JSON object: {{"clusters": list of {{"nodes": list of node JSON strings, "confidence": float (0-1), "reason": str}}}}'
    )
    response = gemini_model.invoke(prompt)
    content = response.content.strip()
    result = extract_json_from_response(content)
    if result is None or "clusters" not in result:
        logger.error(f"Failed to cluster nodes: {content[:500]}")
        return {"clusters": [{"nodes": [node.model_dump_json()], "confidence": 0.0, "reason": "Invalid response"} for node in nodes]}
    return result

def merge_nodes(nodes_to_merge: List[BaseNode]) -> BaseNode:
    """Merge a list of nodes into a single node (unchanged)."""
    if not nodes_to_merge:
        raise ValueError("No nodes provided to merge")
    if len(nodes_to_merge) == 1:
        return nodes_to_merge[0]

    print(f"Merging {len(nodes_to_merge)} nodes: {[n.id for n in nodes_to_merge]}")
    base_node = nodes_to_merge[0]
    merged = BaseNode(
        id=base_node.id,
        type=base_node.type,
        properties=base_node.properties.copy(),
        provenance=NodeProvenance(
            chunk_ids=base_node.provenance.chunk_ids.copy(),
            extraction_timestamp=base_node.provenance.extraction_timestamp,
            confidence_score=base_node.provenance.confidence_score or 0
        )
    )
    for node in nodes_to_merge[1:]:
        for key, value in node.properties.items():
            if value and (key not in merged.properties or not merged.properties[key]):
                merged.properties[key] = value
        merged.provenance.chunk_ids.extend(node.provenance.chunk_ids)
        merged.provenance.confidence_score = max(merged.provenance.confidence_score, node.provenance.confidence_score or 0)
    merged.provenance.chunk_ids = list(set(merged.provenance.chunk_ids))
    return merged

def compare_edges(edges: List[RelationshipInstance], ontology: Dict[str, Any], context: str) -> Dict[str, List[Dict]]:
    """Compare all edges of the same type in one shot using Gemini."""
    if not edges or len(edges) <= 1:
        return {"clusters": [{"edges": [edge.model_dump_json() for edge in edges], "confidence": 1.0, "reason": "Single or no edges"}]}

    edge_type = edges[0].type
    if not all(edge.type == edge_type for edge in edges):
        raise ValueError("All edges must be of the same type for one-shot comparison")

    print(f"Comparing {len(edges)} edges of type {edge_type} in one shot")
    prompt = (
        f"Given the ontology and context, group these edges of type '{edge_type}' into clusters where edges in each cluster represent the same relationship.\n"
        f"Ontology: {json.dumps(ontology)}\n"
        f"Edges: {json.dumps([edge.model_dump_json() for edge in edges], indent=2)}\n"
        f"Context: {context[:10000]}\n"
        f'Return only a JSON object: {{"clusters": list of {{"edges": list of edge JSON strings, "confidence": float (0-1), "reason": str}}}}'
    )
    response = gemini_model.invoke(prompt)
    content = response.content.strip()
    result = extract_json_from_response(content)
    if result is None or "clusters" not in result:
        logger.error(f"Failed to cluster edges: {content[:500]}")
        return {"clusters": [{"edges": [edge.model_dump_json()], "confidence": 0.0, "reason": "Invalid response"} for edge in edges]}
    return result

def merge_edges(edges_to_merge: List[RelationshipInstance]) -> RelationshipInstance:
    """Merge a list of edges into a single edge."""
    if not edges_to_merge:
        raise ValueError("No edges provided to merge")
    if len(edges_to_merge) == 1:
        return edges_to_merge[0]

    print(f"Merging {len(edges_to_merge)} edges: {[e.id for e in edges_to_merge]}")
    base_edge = edges_to_merge[0]
    merged = RelationshipInstance(
        id=base_edge.id,
        type=base_edge.type,
        source_id=base_edge.source_id,
        target_id=base_edge.target_id,
        source_type=base_edge.source_type,
        target_type=base_edge.target_type,
        properties=base_edge.properties.copy(),
        provenance=NodeProvenance(
            chunk_ids=base_edge.provenance.chunk_ids.copy(),
            extraction_timestamp=base_edge.provenance.extraction_timestamp,
            confidence_score=base_edge.provenance.confidence_score or 0
        )
    )
    for edge in edges_to_merge[1:]:
        for key, value in edge.properties.items():
            if value and (key not in merged.properties or not merged.properties[key]):
                merged.properties[key] = value
        merged.provenance.chunk_ids.extend(edge.provenance.chunk_ids)
        merged.provenance.confidence_score = max(merged.provenance.confidence_score, edge.provenance.confidence_score or 0)
    merged.provenance.chunk_ids = list(set(merged.provenance.chunk_ids))
    return merged

def infer_relationship(nodes: List[BaseNode], relationships: List[RelationshipInstance], ontology: Dict[str, Any], context: str) -> List[Dict]:
    """Infer relationships (unchanged)."""
    if not nodes or len(nodes) < 2:
        return []

    connected_nodes = set()
    for rel in relationships:
        connected_nodes.add(rel.source_id)
        connected_nodes.add(rel.target_id)
    
    orphan_nodes = [node for node in nodes if node.id not in connected_nodes]
    if not orphan_nodes:
        print("No orphan nodes found for relationship inference")
        return []

    print(f"Inferring relationships for {len(orphan_nodes)} orphan nodes")
    prompt = (
        f"Given the ontology, context, and current graph, infer meaningful relationships between orphan nodes and other nodes or subgraphs.\n"
        f"Ontology: {json.dumps(ontology)}\n"
        f"Orphan Nodes: {json.dumps([node.model_dump_json() for node in orphan_nodes], indent=2)}\n"
        f"All Nodes: {json.dumps([node.model_dump_json() for node in nodes], indent=2)}\n"
        f"Existing Relationships: {json.dumps([rel.model_dump_json() for rel in relationships], indent=2)}\n"
        f"Context: {context[:10000]}\n"
        f"Return only a JSON object: {{'relationships': list of {{'type': str, 'source': str, 'target': str, 'properties': dict, 'confidence': float (0-1), 'evidence': str}}}}"
    )
    response = gemini_model.invoke(prompt)
    content = response.content.strip()
    result = extract_json_from_response(content)
    if result is None or "relationships" not in result:
        logger.error(f"Failed to infer relationships: {content[:500]}")
        return []
    print(f"Inferred {len(result['relationships'])} relationships")
    return result["relationships"]

def validate_component(components: List[BaseModel], ontology: Dict[str, Any], context: str) -> List[Dict]:
    """Validate a list of nodes or edges of the same type in one shot using Gemini."""
    if not components:
        return []

    # Check if all components are of the same type (node or edge)
    is_node = isinstance(components[0], BaseNode)
    if not all((isinstance(c, BaseNode) == is_node) for c in components):
        raise ValueError("All components must be either nodes or edges, not mixed")

    component_type = "nodes" if is_node else "edges"
    component_count = len(components)
    print(f"Validating {component_count} {component_type} in one shot")

    # Serialize components
    components_json = [c.model_dump_json() for c in components]

    prompt = (
        f"Validate these {component_type} against the ontology and context.\n"
        f"Ontology: {json.dumps(ontology)}\n"
        f"{component_type.capitalize()}: {json.dumps(components_json, indent=2)}\n"
        f"Context: {context[:10000]}\n"
        f'Return only a JSON object: {{"validations": list of {{"component": str (original JSON string), "is_valid": bool, "fixes": dict (specific corrections only, no placeholders)}}}}'
    )
    response = gemini_model.invoke(prompt)
    content = response.content.strip()
    result = extract_json_from_response(content)

    if result is None or "validations" not in result:
        logger.error(f"Failed to validate {component_type}: {content[:500]}")
        return [{"component": c_json, "is_valid": False, "fixes": {}} for c_json in components_json]

    # Ensure the number of validations matches the input
    if len(result["validations"]) != component_count:
        logger.warning(f"Validation count mismatch: expected {component_count}, got {len(result['validations'])}")
        # Fallback to invalid for missing components
        validated_components = {v["component"]: v for v in result["validations"]}
        return [
            validated_components.get(c_json, {"component": c_json, "is_valid": False, "fixes": {}, "reason": "Missing validation"})
            for c_json in components_json
        ]

    return result["validations"]

# Handoff Tools (unchanged)
transfer_to_resolution = create_handoff_tool(agent_name="resolution_agent", description="Transfer to entity resolution agent.")
transfer_to_inference = create_handoff_tool(agent_name="inference_agent", description="Transfer to relationship inference agent.")
transfer_to_validation = create_handoff_tool(agent_name="validation_agent", description="Transfer to validation agent.")

# Prompt Factory (unchanged)
def make_prompt(base_prompt: str) -> Callable[[GraphState, RunnableConfig], List]:
    def prompt(state: GraphState, config: RunnableConfig) -> List:
        graph_dict = {"nodes": [n.model_dump() for n in state.graph.nodes], "relationships": [r.model_dump() for r in state.graph.relationships]}
        system_prompt = (
            base_prompt + "\nAct autonomously, reasoning about entities and relationships using context and ontology." +
            f"\nOntology: {json.dumps(state.ontology, indent=2)}" + f"\nCurrent graph: {json.dumps(graph_dict, indent=2)}" +
            f"\nContext: {state.context[:1000]}"
        )
        return [{"role": "system", "content": system_prompt}] + state.messages
    return prompt

# Define Agents (unchanged)
extraction_agent = create_react_agent(gemini_model, tools=[extract_initial_graph, transfer_to_resolution], prompt=make_prompt("Extract an initial graph from PDFs using the ontology."), name="extraction_agent")
resolution_agent = create_react_agent(gemini_model, tools=[compare_nodes, merge_nodes, compare_edges, merge_edges, transfer_to_inference], prompt=make_prompt("Resolve ambiguous nodes and edges."), name="resolution_agent")
inference_agent = create_react_agent(gemini_model, tools=[infer_relationship, transfer_to_validation], prompt=make_prompt("Infer relationships between nodes."), name="inference_agent")
validation_agent = create_react_agent(gemini_model, tools=[validate_component, transfer_to_resolution], prompt=make_prompt("Validate and refine the graph."), name="validation_agent")