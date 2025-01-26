from typing import List, Tuple
from neo4j import GraphDatabase
from app.schemas.global_merge import DbNode, DbEdge
from app.config import settings

class StagingGraphExtractor:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.UID_FIELD = '_uid_'

    def get_subgraph(self, label: str) -> tuple[List[DbNode], List[DbEdge]]:
        with self.driver.session() as session:
            # Node query remains the same
            nodes_query = f"""
            MATCH (n:{label})
            CALL apoc.path.subgraphNodes(n, {{
                relationshipFilter: '>',
                maxLevel: 1000
            }})
            YIELD node
            WITH DISTINCT node
            RETURN collect({{node: node, {self.UID_FIELD}: node.{self.UID_FIELD}}}) as nodeWithIds
            """

            # Modified relationship query to handle missing _uid_
            rels_query = f"""
            MATCH (n:{label})
            CALL apoc.path.subgraphAll(n, {{
                relationshipFilter: '>',
                maxLevel: 1000
            }})
            YIELD relationships
            UNWIND relationships as rel
            WITH DISTINCT rel,
                 CASE
                     WHEN rel.{self.UID_FIELD} IS NOT NULL
                     THEN rel.{self.UID_FIELD}
                     ELSE elementId(rel) + '-rel'
                 END as rel_id
            RETURN collect({{
                rel: rel,
                {self.UID_FIELD}: rel_id,
                start: startNode(rel).{self.UID_FIELD},
                end: endNode(rel).{self.UID_FIELD}
            }}) as relationshipWithIds
            """

            nodes_result = session.run(nodes_query, label=label)
            nodes_data = nodes_result.single()["nodeWithIds"]

            rels_result = session.run(rels_query, label=label)
            rels_data = rels_result.single()["relationshipWithIds"]

            # Use dict to ensure uniqueness by ID
            nodes_dict = {
                n[self.UID_FIELD]: DbNode(
                    id=n[self.UID_FIELD],
                    labels=[l for l in list(n['node'].labels) if l != label],
                    properties={k: v for k, v in dict(n['node']).items() if k != self.UID_FIELD},
                )
                for n in nodes_data
            }

            edges = [
                DbEdge(
                    id=r[self.UID_FIELD],
                    type=r['rel'].type,
                    properties={k: v for k, v in dict(r['rel']).items() if k != self.UID_FIELD},
                    source_id=r['start'],
                    target_id=r['end']
                )
                for r in rels_data
            ]

            return list(nodes_dict.values()), edges

    def validate_subgraph(self, nodes: List[DbNode], edges: List[DbEdge]) -> bool:
        """
        Validate that the subgraph is properly connected
        """
        # Create set of all node IDs
        node_ids = {node.id for node in nodes}

        # Check if all edge endpoints exist in nodes
        for edge in edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                return False

        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.close()
        
def get_subgraph(label: str) -> Tuple[List[DbNode], List[DbEdge]]:
  with StagingGraphExtractor(settings.STAGING_NEO4J_URI, settings.STAGING_NEO4J_USER, 
                             settings.STAGING_NEO4J_PASSWORD) as extractor:
    return extractor.get_subgraph(label=label)