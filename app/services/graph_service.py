import traceback
from typing import Dict, Any
from neo4j import GraphDatabase
from app.schemas.graph import Node, Edge, GraphResponse
from app.schemas.graph_changes import (
    NodeCreation, NodeUpdate, EdgeCreation, EdgeUpdate,
    SaveGraphRequest, SaveGraphResponse, Message
)
from app.utils.logger import logger
from uuid import uuid4
from app.utils.constants import TRANSFORM_ID, MERGE_ID

class GraphService:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def get_graph_by_transform_id(self, transform_id: str, limit: int = 1000, skip: int = 0) -> GraphResponse:
        """
        Retrieve nodes by transform ID and their relationships
        
        Args:
            transform_id: ID of the transform
            limit: Maximum number of nodes to return
            skip: Number of nodes to skip (for pagination)
            
        Returns:
            GraphResponse containing nodes and edges
        """
        try:
            # First get total counts
            count_query = f"""
            MATCH (n)
            WHERE n.{TRANSFORM_ID} = '{transform_id}'
            WITH count(n) as node_count
            OPTIONAL MATCH (n)-[r]-()
            RETURN node_count, count(DISTINCT r) as edge_count
            """

            with self.driver.session() as session:
                count_result = session.run(count_query)
                count_data = count_result.single()
                total_nodes = count_data["node_count"]
                total_edges = count_data["edge_count"]

                # Now get the actual data with pagination
                query = f"""
                MATCH (n)
                WHERE n.{TRANSFORM_ID} = '{transform_id}'
                WITH n ORDER BY n.id
                SKIP {skip} LIMIT {limit}
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN 
                    collect(DISTINCT n) as nodes,
                    collect(DISTINCT r) as relationships,
                    collect(DISTINCT m) as connected_nodes
                """

                result = session.run(query)
                data = result.single()

                # Transform results
                nodes_list = []
                edges_list = []
                seen_nodes = set()
                seen_edges = set()

                def get_actual_label(node_labels):
                    return list(node_labels)[0]

                def extract_properties(entity):
                    """Extract properties from node/relationship, excluding special fields"""
                    props = {}
                    entity_dict = dict(entity)
                    
                    # Get all properties that start with prop_
                    for key, value in entity_dict.items():
                        if isinstance(value, str):
                                try:
                                    if value.startswith('[') or value.startswith('{'):
                                        value = eval(value)
                                except:
                                    pass
                        props[key] = value
                    return props

                # Process main nodes
                for node in data["nodes"]:
                    node_id = node.get("id")
                    if node_id and node_id not in seen_nodes:
                        actual_label = get_actual_label(node.labels)
                        node_props = extract_properties(node)
                        nodes_list.append(Node(
                            id=node_id,
                            label=actual_label,
                            properties=node_props,
                            type=actual_label
                        ))
                        seen_nodes.add(node_id)

                # Process connected nodes
                for node in data["connected_nodes"]:
                    if node is not None:
                        node_id = node.get("id")
                        if node_id and node_id not in seen_nodes:
                            actual_label = get_actual_label(node.labels)
                            node_props = extract_properties(node)
                            nodes_list.append(Node(
                                id=node_id,
                                label=actual_label,
                                properties=node_props,
                                type=actual_label
                            ))
                            seen_nodes.add(node_id)

                # Process relationships
                for rel in data["relationships"]:
                    if rel is not None:
                        edge_id = rel.get("id", str(rel.id))
                        if edge_id not in seen_edges:
                            source_id = rel.start_node.get("id")
                            target_id = rel.end_node.get("id")
                            if source_id and target_id:
                                edge_props = extract_properties(rel)
                                edges_list.append(Edge(
                                    id=edge_id,
                                    source=source_id,
                                    target=target_id,
                                    type=str(rel.type),
                                    properties=edge_props
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

    def _flatten_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten properties for Neo4j storage"""
        flattened = {}
        for key, value in properties.items():
            # Skip null values and internal fields
            if value is not None and not key.startswith('_') and key != 'id':
                # Add prop_prop_ prefix to avoid conflicts with reserved fields
                # prop_key = f"prop_{key}"
                # Convert non-primitive types to string
                if isinstance(value, (dict, list)):
                    value = str(value)
                flattened[key] = value
        return flattened

    def create_node(self, tx, node: NodeCreation, transform_id: str):
        """Create a new node"""
        # Flatten properties
        props = self._flatten_properties(node.properties)
        # Build dynamic SET clause
        set_clauses = [f"n.{key} = ${key}" for key in props.keys()]
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        CREATE (n:`{transform_id}`:`{node.label}`)
        SET n.id = $id, n.type = $type
        """
        if set_clause:
            query += f", {set_clause}"

        # Prepare parameters
        params = {
            "id": str(uuid4()),
            "type": node.type,
            **props
        }
        
        tx.run(query, params)

    def update_node(self, tx, node: NodeUpdate, transform_id: str):
        """Update an existing node"""
        # First get existing properties
        result = tx.run(
            f"""
            MATCH (n)
            WHERE n.{TRANSFORM_ID} = "{transform_id}" AND
            n.id = $id
            RETURN n
            """,
            id=node.id
        ).single()
        
        if not result:
            return
            
        existing_node = result['n']
        existing_props = {k: v for k, v in dict(existing_node).items() 
                        if not k.startswith('_') and k != 'type' and k != 'id'}
        
        # Get new properties
        new_props = self._flatten_properties(node.properties)
        
        # Build REMOVE clause for properties not in new set
        remove_props = set(existing_props.keys()) - set(new_props.keys())
        remove_clause = ""
        if remove_props:
            remove_clause = "REMOVE " + ", ".join(f"n.{prop}" for prop in remove_props)
        
        # Build SET clause for new/updated properties
        set_clauses = [f"n.{key} = ${key}" for key in new_props.keys()]
        set_clause = ""
        if set_clauses:
            set_clause = "SET " + ", ".join(set_clauses)
        
        # Build and execute query
        query_parts = [
            f"""MATCH (n)
            WHERE n.{TRANSFORM_ID} = "{transform_id}" AND
            n.id = $id
            """
        ]
        if set_clause:
            query_parts.append(set_clause)
        if remove_clause:
            query_parts.append(remove_clause)
            
        query = "\n".join(query_parts)
        
        # Execute update if we have changes
        print(new_props)
        if set_clause or remove_clause:
            tx.run(
                query,
                id=node.id,
                **new_props
            )

    def delete_node(self, tx, node_id: str, transform_id: str):
        """Delete a node"""
        tx.run(
            f"""
            MATCH (n)
            WHERE n.{TRANSFORM_ID} = "{transform_id}" AND
            n.id = $id
            DETACH DELETE n
            """,
            id=node_id
        )

    def create_edge(self, tx, edge: EdgeCreation, transform_id: str):
        """Create a new edge"""
        # Flatten properties
        props = self._flatten_properties(edge.properties)
        # Build dynamic SET clause
        set_clauses = [f"r.{key} = ${key}" for key in props.keys()]
        set_clauses.append("r.id = $id")
        set_clauses.append("r.type = $type")
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        MATCH (source:`{transform_id}` {{id: $source_id}})
        MATCH (target:`{transform_id}` {{id: $target_id}})
        CREATE (source)-[r:`{edge.type}`]->(target)
        SET {set_clause}
        """
        
        # Prepare parameters
        params = {
            "id": str(uuid4()),
            "source_id": edge.source,
            "target_id": edge.target,
            "type": edge.type,
            **props
        }
        
        tx.run(query, params)

    def update_edge(self, tx, edge: EdgeUpdate, transform_id: str):
        """Update an existing edge"""
        # First get existing properties
        result = tx.run(
            """
            MATCH ()-[r]->()
            WHERE r.id = $id
            RETURN r
            """,
            id=edge.id
        ).single()
        
        if not result:
            return
            
        existing_edge = result['r']
        existing_props = {k: v for k, v in dict(existing_edge).items() 
                        if not k.startswith('_') and k != 'type' and k != 'id'}
        
        # Get new properties
        new_props = self._flatten_properties(edge.properties)
        
        # Build REMOVE clause for properties not in new set
        remove_props = set(existing_props.keys()) - set(new_props.keys())
        remove_clause = ""
        if remove_props:
            remove_clause = "REMOVE " + ", ".join(f"r.{prop}" for prop in remove_props)
        
        # Build SET clause for new/updated properties
        set_clauses = [f"r.{key} = ${key}" for key in new_props.keys()]
        set_clause = ""
        if set_clauses:
            set_clause = "SET " + ", ".join(set_clauses)
        
        # Build and execute query
        query_parts = [
            "MATCH ()-[r]->() WHERE r.id = $id"
        ]
        if set_clause:
            query_parts.append(set_clause)
        if remove_clause:
            query_parts.append(remove_clause)
            
        query = "\n".join(query_parts)
        
        # Execute update if we have changes
        if set_clause or remove_clause:
            tx.run(
                query,
                id=edge.id,
                **new_props
            )

    def delete_edge(self, tx, edge_id: str):
        """Delete an edge"""
        tx.run(
            """
            MATCH ()-[r]->()
            WHERE r.id = $id
            DELETE r
            """,
            id=edge_id
        )

    def save_graph_changes(self, transform_id: str, changes: SaveGraphRequest) -> SaveGraphResponse:
        """Save graph changes in a single transaction"""
        messages = []
        
        with self.driver.session() as session:
            def inner_save(tx):
                # Apply changes in order
                # 1. Create new nodes
                if changes.nodes:
                    for node in changes.nodes.created:
                        self.create_node(tx, node, transform_id)

                    # 2. Update existing nodes
                    for node in changes.nodes.updated:
                        self.update_node(tx, node, transform_id)

                # 3. Create new edges
                if changes.edges:
                    for edge in changes.edges.created:
                        try:
                            self.create_edge(tx, edge, transform_id)
                        except Exception as e:
                            messages.append(Message(
                                type='warning',
                                message=f"Failed to create edge {edge.id}: {str(e)}"
                            ))

                    # 4. Update existing edges
                    for edge in changes.edges.updated:
                        self.update_edge(tx, edge, transform_id)

                    # 5. Delete edges
                    for edge_id in changes.edges.deleted:
                        self.delete_edge(tx, edge_id)

                # 6. Delete nodes
                if changes.nodes:
                    for node_id in changes.nodes.deleted:
                        self.delete_node(tx, node_id, transform_id)

            try:
                # Execute transaction
                session.execute_write(inner_save)

                # Get updated graph state
                updated_graph = self.get_graph_by_transform_id(transform_id)

                # Convert Node and Edge objects to dictionaries
                nodes_dict = [
                    {
                        "id": node.id,
                        "label": node.label,
                        "type": node.type,
                        "properties": node.properties
                    } for node in updated_graph.nodes
                ]
                
                edges_dict = [
                    {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.type,
                        "properties": edge.properties
                    } for edge in updated_graph.edges
                ]

                return SaveGraphResponse(
                    data={"nodes": nodes_dict, "edges": edges_dict},
                    messages=messages if messages else None
                )

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error saving graph changes: {str(e)}")
                raise

    def __del__(self):
        """Cleanup"""
        self.close()
