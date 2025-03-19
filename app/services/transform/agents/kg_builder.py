
from app.services.transform.agent import extraction_agent, resolution_agent, inference_agent, validation_agent
from app.services.transform.agents.ontology import OntologyParser
from app.services.transform.agent import GraphState
from app.services.transform.models import DocumentKnowledgeGraph
from langgraph_swarm import create_swarm
from langgraph.checkpoint.memory import InMemorySaver
from typing import List, Optional, Callable
from datetime import datetime, timezone
import uuid
from app.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    ExtractionMetrics
)

class KnowledgeGraphBuilder:
    def __init__(self, ontology_yaml: str):
        self.ontology_parser = OntologyParser(ontology_yaml)
        self.checkpointer = InMemorySaver()
        self.app = self._build_workflow()

    def _build_workflow(self):
        builder = create_swarm(
            agents=[extraction_agent, resolution_agent, inference_agent, validation_agent],
            default_active_agent="extraction_agent"
        )
        return builder.compile(checkpointer=self.checkpointer)

    async def build_graph_from_pdfs(
        self,
        pdf_paths: List[str],
        transform_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DocumentKnowledgeGraph:
        """Build a knowledge graph from PDFs."""
        state = GraphState(
            pdf_paths=pdf_paths,
            ontology=self.ontology_parser.parsed_ontology,
            ontology_yaml=self.ontology_parser.ontology_yaml,
            entities_only_model=self.ontology_parser.build_entities_only_model(),  # Updated to call method
            relationships_only_model=self.ontology_parser.build_relationships_only_model(),  # Updated to call method
            messages=[{"role": "user", "content": "Start graph extraction from PDFs."}]
        )
        config = {"configurable": {"thread_id": transform_id, "user_id": "1"}}

        # Step 1: Extraction
        initial_graph = extract_initial_graph(state)
        print(initial_graph)
        state.graph.nodes = initial_graph.nodes
        state.graph.relationships = initial_graph.relationships

        # Step 2: Resolution
        # Node resolution (unchanged)
        nodes = state.graph.nodes
        nodes_by_type = {}
        for node in nodes:
            nodes_by_type.setdefault(node.type, []).append(node)

        resolved_nodes = []
        for node_type, type_nodes in nodes_by_type.items():
            clustering_result = compare_nodes(type_nodes, self.ontology_parser.parsed_ontology, state.context)
            for cluster in clustering_result["clusters"]:
                cluster_nodes = [BaseNode.model_validate_json(node_json) for node_json in cluster["nodes"]]
                if len(cluster_nodes) > 1:
                    merged_node = merge_nodes(cluster_nodes)
                    resolved_nodes.append(merged_node)
                    state.decision_log.append(f"Merged {len(cluster_nodes)} nodes of type {node_type} into {merged_node.id}: {cluster['reason']}")
                    state.confidence_scores[f"node_merge_{merged_node.id}"] = cluster["confidence"]
                else:
                    resolved_nodes.append(cluster_nodes[0])
                    state.decision_log.append(f"Kept single node {cluster_nodes[0].id} of type {node_type}: {cluster['reason']}")
                    state.confidence_scores[f"node_merge_{cluster_nodes[0].id}"] = cluster["confidence"]
        state.graph.nodes = resolved_nodes

        # Edge resolution (optimized)
        edges = state.graph.relationships
        edges_by_type = {}
        for edge in edges:
            edges_by_type.setdefault(edge.type, []).append(edge)

        resolved_edges = []
        for edge_type, type_edges in edges_by_type.items():
            clustering_result = compare_edges(type_edges, self.ontology_parser.parsed_ontology, state.context)
            for cluster in clustering_result["clusters"]:
                cluster_edges = [RelationshipInstance.model_validate_json(edge_json) for edge_json in cluster["edges"]]
                if len(cluster_edges) > 1:
                    merged_edge = merge_edges(cluster_edges)
                    resolved_edges.append(merged_edge)
                    state.decision_log.append(f"Merged {len(cluster_edges)} edges of type {edge_type} into {merged_edge.id}: {cluster['reason']}")
                    state.confidence_scores[f"edge_merge_{merged_edge.id}"] = cluster["confidence"]
                else:
                    resolved_edges.append(cluster_edges[0])
                    state.decision_log.append(f"Kept single edge {cluster_edges[0].id} of type {edge_type}: {cluster['reason']}")
                    state.confidence_scores[f"edge_merge_{cluster_edges[0].id}"] = cluster["confidence"]
        state.graph.relationships = resolved_edges

        # Step 3: Inference with Deduplication
        seen_rels = set()
        inferred_rels = infer_relationship(state.graph.nodes, state.graph.relationships, self.ontology_parser.parsed_ontology, state.context)
        for rel in inferred_rels:
            if rel.get("type") and rel.get("source") and rel.get("target"):
                rel_key = (rel["type"], rel["source"], rel["target"])
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    source_node = next(node for node in state.graph.nodes if node.id == rel["source"])
                    target_node = next(node for node in state.graph.nodes if node.id == rel["target"])
                    state.graph.relationships.append(RelationshipInstance(
                        id=str(uuid.uuid4()),
                        type=rel["type"],
                        source_id=rel["source"],
                        target_id=rel["target"],
                        source_type=source_node.type,
                        target_type=target_node.type,
                        properties=rel["properties"],
                        provenance=NodeProvenance(
                            chunk_ids=[f"{transform_id}_0"],
                            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                            confidence_score=rel["confidence"]
                        )
                    ))
                    state.decision_log.append(f"Inferred {rel['type']} between {rel['source']} and {rel['target']} with evidence: {rel['evidence']}")
                    state.confidence_scores[f"rel_{rel['source']}_{rel['target']}"] = rel["confidence"]

        # Step 4: Validation
        for node in state.graph.nodes:
            validation = validate_component(node, self.ontology_parser.parsed_ontology, state.context)
            if not validation["is_valid"]:
                node.properties.update(validation["fixes"])
                state.decision_log.append(f"Fixed node {node.id}: {validation['fixes']}")
        valid_edges = []
        for edge in state.graph.relationships:
            validation = validate_component(edge, self.ontology_parser.parsed_ontology, state.context)
            if validation["is_valid"]:
                fixes = {k: v for k, v in validation["fixes"].items() if k in edge.properties}
                edge.properties.update(fixes)
                valid_edges.append(edge)
            else:
                state.decision_log.append(f"Removed invalid edge {edge.source_id}->{edge.target_id}")
        state.graph.relationships = valid_edges

        # Finalize metrics
        state.graph.metrics = ExtractionMetrics(
            start_time=datetime.now(timezone.utc),
            total_nodes=len(state.graph.nodes),
            total_relationships=len(state.graph.relationships),
            merged_nodes=state.graph.metrics.merged_nodes if state.graph.metrics else 0
        )
        state.graph.metrics.finalize()
        state.graph.confidence_score = sum(state.confidence_scores.values()) / len(state.confidence_scores) if state.confidence_scores else 0.8

        if progress_callback:
            await progress_callback(len(pdf_paths), len(pdf_paths)) if asyncio.iscoroutinefunction(progress_callback) else progress_callback(len(pdf_paths), len(pdf_paths))

        logger.info(f"Graph built: {state.graph.metrics}")
        logger.info(f"Decisions: {state.decision_log}")
        return state.graph