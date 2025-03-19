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

project_id = "graphit-sandbox"
location = "us-central1"
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(google.auth.transport.requests.Request())

gemini_model = ChatOpenAI(
    base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
    api_key=credentials.token,
    model='google/gemini-2.0-flash-lite-001'
)


# Custom JSON encoder for Pydantic models
def custom_json_encoder(obj):
    if isinstance(obj, (BaseNode, RelationshipInstance, NodeProvenance, ExtractionMetrics, DocumentKnowledgeGraph)):
        return obj.dict()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# State to track graph construction
class GraphState(BaseModel):
    pdf_paths: List[str] = Field(default_factory=list)
    ontology: Dict[str, Any] = Field(default_factory=dict)
    ontology_model: Optional[Type[BaseModel]] = None
    graph: DocumentKnowledgeGraph = Field(default_factory=DocumentKnowledgeGraph)
    context: str = Field(default="")
    decision_log: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    messages: List[Dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

# Helper to extract JSON from response
def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """Extract JSON from a response that may contain explanatory text."""
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
    result = extract_structured_data(state.pdf_paths[0], parsed_ontology=state.ontology, model=state.ontology_model)
    print(result)
    state.decision_log.append("Initial graph extracted from PDFs")
    return result

def compare_nodes(nodes: List[BaseNode], ontology: BaseModel, context: str) -> Dict[str, List[Dict]]:
    """Compare all nodes of the same type in one shot using Gemini."""
    print(f"Comparing {len(nodes)} nodes of type {nodes[0].type}")
    if not nodes or len(nodes) <= 1:
        return {"clusters": [{"nodes": [node.model_dump_json() for node in nodes], "confidence": 1.0, "reason": "Single or no nodes"}]}

    # Group nodes by type first (assuming this is called per type)
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
    """Merge a list of nodes into a single node."""
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

def compare_edges(edge1: RelationshipInstance, edge2: RelationshipInstance, ontology: BaseModel, context: str) -> RelationshipInstance:
    """Use Gemini to decide if two edges should merge."""
    if edge1.type != edge2.type:
        return {"should_merge": False, "confidence": 1.0, "reason": "Different edge types"}
    print(f"Comparing edges: {edge1} and {edge2}")
    prompt = (
        f"Given the ontology and context, determine if these edges represent the same relationship.\n"
        f"Ontology: {json.dumps(ontology)}\n"
        f"Edge 1: {edge1.model_dump_json()}\n"
        f"Edge 2: {edge2.model_dump_json()}\n"
        f"Context: {context[:10000]}\n"
        f'Return only a JSON object: {{"should_merge": bool, "confidence": float (0-1), "reason": str}}'
    )
    response = gemini_model.invoke(prompt)
    content = response.content.strip()
    result = extract_json_from_response(content)
    if result is None:
        return {"should_merge": False, "confidence": 0.0, "reason": "Invalid response from Gemini"}
    return result

def merge_edges(edge1: RelationshipInstance, edge2: RelationshipInstance) -> RelationshipInstance:
    """Merge two edges, combining properties and provenance."""
    print(f"Merging edges: {edge1} and {edge2}")
    merged = RelationshipInstance(
        id=edge1.id,
        type=edge1.type,
        source_id=edge1.source_id,
        target_id=edge1.target_id,
        source_type=edge1.source_type,
        target_type=edge1.target_type,
        properties=edge1.properties.copy(),
        provenance=NodeProvenance(
            chunk_ids=edge1.provenance.chunk_ids.copy(),
            extraction_timestamp=edge1.provenance.extraction_timestamp,
            confidence_score=max(edge1.provenance.confidence_score or 0, edge2.provenance.confidence_score or 0)
        )
    )
    for key, value in edge2.properties.items():
        if value and (key not in merged.properties or not merged.properties[key]):
            merged.properties[key] = value
    merged.provenance.chunk_ids.extend(edge2.provenance.chunk_ids)
    merged.provenance.chunk_ids = list(set(merged.provenance.chunk_ids))
    return merged

def infer_relationship(nodes: List[BaseNode], relationships: List[RelationshipInstance], ontology: BaseModel, context: str) -> List[Dict]:
    """Infer relationships for orphan nodes and disjointed subgraphs in one shot using Gemini."""
    if not nodes or len(nodes) < 2:
        return []

    # Identify orphan nodes and disjointed subgraphs
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
        f"Given the ontology, context, and current graph, infer meaningful relationships between orphan nodes (nodes with no current relationships) and other nodes or subgraphs.\n"
        f"Ontology: {json.dumps(ontology)}\n"
        f"Orphan Nodes: {json.dumps([node.model_dump_json() for node in orphan_nodes], indent=2)}\n"
        f"All Nodes: {json.dumps([node.model_dump_json() for node in nodes], indent=2)}\n"
        f"Existing Relationships: {json.dumps([rel.model_dump_json() for rel in relationships], indent=2)}\n"
        f"Context: {context[:10000]}\n"
        f"Return only a JSON object: {{'relationships': list of {{'type': str (valid from ontology), 'source': str (node id), 'target': str (node id), 'properties': dict (specific details with evidence from context), 'confidence': float (0-1), 'evidence': str (text from context justifying the relationship)}}}}"
    )
    response = gemini_model.invoke(prompt)
    content = response.content.strip()
    result = extract_json_from_response(content)
    if result is None or "relationships" not in result:
        logger.error(f"Failed to infer relationships: {content[:500]}")
        return []
    print(f"Inferred {len(result['relationships'])} relationships")
    return result["relationships"]

def validate_component(component: BaseModel, ontology: BaseModel, context: str) -> Dict:
    """Use Gemini to validate nodes or edges without placeholders."""
    print(f"Validating component: {component}")
    component_serializable = component.model_dump_json()
    is_node = "properties" in component_serializable
    
    prompt = (
        f"Validate this {'node' if is_node else 'edge'} against the ontology and context.\n"
        f"Ontology: {json.dumps(ontology)}\n"
        f"{'Node' if is_node else 'Edge'}: {json.dumps(component_serializable, indent=2)}\n"
        f"Context: {context[:10000]}\n"
        f'Return only a JSON object: {{"is_valid": bool, "fixes": dict (specific corrections only, no placeholders or messages)}}'
    )
    response = gemini_model.invoke(prompt)
    content = response.content.strip()
    result = extract_json_from_response(content)
    if result is None:
        return {"is_valid": False, "fixes": {}}
    return result

# Handoff Tools
transfer_to_resolution = create_handoff_tool(
    agent_name="resolution_agent",
    description="Transfer to entity resolution agent."
)
transfer_to_inference = create_handoff_tool(
    agent_name="inference_agent",
    description="Transfer to relationship inference agent."
)
transfer_to_validation = create_handoff_tool(
    agent_name="validation_agent",
    description="Transfer to validation agent."
)

# Prompt Factory
def make_prompt(base_prompt: str) -> Callable[[GraphState, RunnableConfig], List]:
    def prompt(state: GraphState, config: RunnableConfig) -> List:
        graph_dict = {
            "nodes": [n.dict() for n in state.graph.nodes],
            "relationships": [r.dict() for r in state.graph.relationships]
        }
        system_prompt = (
            base_prompt +
            "\nAct autonomously, reasoning about entities and relationships using context and ontology." +
            f"\nOntology: {json.dumps(state.ontology, indent=2)}" +
            f"\nCurrent graph: {json.dumps(graph_dict, indent=2)}" +
            f"\nContext: {state.context[:1000]}"
        )
        return [{"role": "system", "content": system_prompt}] + state.messages
    return prompt

# Define Agents
extraction_agent = create_react_agent(
    gemini_model,
    tools=[extract_initial_graph, transfer_to_resolution],
    prompt=make_prompt("Extract an initial graph from PDFs using the ontology."),
    name="extraction_agent"
)

resolution_agent = create_react_agent(
    gemini_model,
    tools=[compare_nodes, merge_nodes, compare_edges, merge_edges, transfer_to_inference],
    prompt=make_prompt("Resolve ambiguous nodes and edges by reasoning about their identity."),
    name="resolution_agent"
)

inference_agent = create_react_agent(
    gemini_model,
    tools=[infer_relationship, transfer_to_validation],
    prompt=make_prompt("Infer relationships between nodes based on context and ontology."),
    name="inference_agent"
)

validation_agent = create_react_agent(
    gemini_model,
    tools=[validate_component, transfer_to_resolution],
    prompt=make_prompt("Validate and refine the graph, fixing inconsistencies."),
    name="validation_agent"
)