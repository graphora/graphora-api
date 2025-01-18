from typing import List, Dict
from schemas.transform import KnowledgeGraph, Node, Edge, Metadata

class KnowledgeGraphManager:
  def merge_graphs(self, graphs: List[KnowledgeGraph]) -> Dict[str, Dict[str, KnowledgeGraph]]:
    section_graphs = {}

    for graph in graphs:
        section = graph.metadata.section
        subsection = '|'.join(sorted(graph.metadata.subsections))

        if section not in section_graphs:
            section_graphs[section] = {}

        if subsection not in section_graphs[section]:
            section_graphs[section][subsection] = graph
            continue

        existing = section_graphs[section][subsection]

        merged_nodes = {
            node.id: Node(
                id=node.id,
                properties=node.properties.copy()
            ) for node in existing.nodes
        }

        for node in graph.nodes:
            if node.id in merged_nodes:
                merged_nodes[node.id].properties.update(node.properties)
            else:
                merged_nodes[node.id] = Node(
                    id=node.id,
                    properties=node.properties.copy()
                )

        edge_map = {
            (e.from_, e.to, e.relationship): Edge.model_validate({
                "from": e.from_,
                "to": e.to,
                "relationship": e.relationship,
                "properties": e.properties.copy()
            }) for e in existing.edges
        }

        for edge in graph.edges:
            key = (edge.from_, edge.to, edge.relationship)
            if key in edge_map:
                edge_map[key].properties.update(edge.properties)
            else:
                edge_map[key] = Edge.model_validate({
                    "from": edge.from_,
                    "to": edge.to,
                    "relationship": edge.relationship,
                    "properties": edge.properties.copy()
                })

        section_graphs[section][subsection] = KnowledgeGraph(
            metadata=Metadata(
                section=section,
                subsections=graph.metadata.subsections
            ),
            nodes=list(merged_nodes.values()),
            edges=list(edge_map.values())
        )

    return section_graphs

  def get_section_graph(self, graphs: Dict[str, Dict[str, KnowledgeGraph]],
                    section: str, subsection: str = None) -> KnowledgeGraph:
    if section not in graphs:
        return None

    if subsection:
        return graphs[section].get(subsection)

    return graphs[section]

  def filter_by_section(self, graphs: Dict[str, KnowledgeGraph], section: str) -> KnowledgeGraph:
    return {k:v for k,v in graphs.items() if section in k}