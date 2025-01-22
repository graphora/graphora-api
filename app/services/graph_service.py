from typing import Dict, List, Any, Tuple
from neo4j import GraphDatabase
from app.schemas.graph import Node, Edge, GraphResponse
from app.utils.logger import logger

class GraphService:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def get_graph_by_label(self, label: str, limit: int = 1000, skip: int = 0) -> GraphResponse:
        """
        Retrieve nodes by label and their relationships
        
        Args:
            label: Node label to query (including batch prefix)
            limit: Maximum number of nodes to return
            skip: Number of nodes to skip (for pagination)
            
        Returns:
            GraphResponse containing nodes and edges
        """
        try:
            # First get total counts
            count_query = """
            MATCH (n:`%s`)
            WITH count(n) as node_count
            OPTIONAL MATCH (n:`%s`)-[r]-()
            RETURN node_count, count(DISTINCT r) as edge_count
            """ % (label, label)

            with self.driver.session() as session:
                count_result = session.run(count_query)
                count_data = count_result.single()
                total_nodes = count_data["node_count"]
                total_edges = count_data["edge_count"]

                # Now get the actual data with pagination
                query = """
                MATCH (n:`%s`)
                WITH n ORDER BY n.id
                SKIP $skip LIMIT $limit
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN 
                    collect(DISTINCT n) as nodes,
                    collect(DISTINCT r) as relationships,
                    collect(DISTINCT m) as connected_nodes
                """ % label

                result = session.run(query, {"limit": limit, "skip": skip})
                data = result.single()

                # Transform results
                nodes_list = []
                edges_list = []
                seen_nodes = set()
                seen_edges = set()

                def get_actual_label(node_labels):
                    """Get the actual node label, ignoring batch labels"""
                    # Convert labels to list and remove the batch label
                    labels = [l for l in node_labels if not l.startswith(label)]
                    # Return the first non-batch label, or the first label if all are batch labels
                    return labels[0] if labels else next(iter(node_labels))

                # Process main nodes
                for node in data["nodes"]:
                    node_id = str(node.id)
                    if node_id not in seen_nodes:
                        actual_label = get_actual_label(node.labels)
                        nodes_list.append(Node(
                            id=node_id,
                            label=actual_label,
                            properties=dict(node.items()),
                            type=actual_label
                        ))
                        seen_nodes.add(node_id)

                # Process connected nodes
                for node in data["connected_nodes"]:
                    if node is not None:
                        node_id = str(node.id)
                        if node_id not in seen_nodes:
                            actual_label = get_actual_label(node.labels)
                            nodes_list.append(Node(
                                id=node_id,
                                label=actual_label,
                                properties=dict(node.items()),
                                type=actual_label
                            ))
                            seen_nodes.add(node_id)

                # Process relationships
                for rel in data["relationships"]:
                    if rel is not None:
                        edge_id = str(rel.id)
                        if edge_id not in seen_edges:
                            edges_list.append(Edge(
                                id=edge_id,
                                source=str(rel.start_node.id),
                                target=str(rel.end_node.id),
                                type=str(rel.type),
                                properties=dict(rel.items())
                            ))
                            seen_edges.add(edge_id)

                return GraphResponse(
                    nodes=nodes_list,
                    edges=edges_list,
                    total_nodes=total_nodes,
                    total_edges=total_edges
                )

        except Exception as e:
            logger.error(f"Error retrieving graph data: {str(e)}")
            raise

    def __del__(self):
        """Cleanup"""
        self.close()
