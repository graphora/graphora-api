from app.schemas.global_merge import DbNode, ERState, NodeResolutionResult, ResolutionStatus
from typing import List, Dict

class DisambiguationAgent:
    """Agent for handling node disambiguation"""

    def __init__(self, prod_db, review_queue, ontology=None):
        self.prod_db = prod_db
        self.review_queue = review_queue
        self.ontology = ontology
        self.UID_FIELD = '_uid_'
        self.IGNORED_PROPERTIES = {'_merged_ids', self.UID_FIELD}

        # Configure similarity thresholds
        self.thresholds = {
            'name_match': 0.85,
            'exact_match': 1.0,
            'strong_match': 0.9,
            'moderate_match': 0.75,
            'weak_match': 0.6,
            'auto_resolve': 0.95
        }

    def find_similar_nodes(self, node: DbNode) -> List[Dict]:
        """Find similar nodes using property matching"""
        # Build base query to match nodes with exact same labels
        labels_clause = ':'.join(node.labels)

        # Filter out ignored properties
        staging_props = {k: v for k, v in node.properties.items()
                       if k not in self.IGNORED_PROPERTIES}

        similarity_calculations = []
        params = {}

        # Add similarity calculation for each property
        for prop, value in staging_props.items():
            param_name = f"prop_{prop}"
            params[param_name] = str(value)
            similarity_calculations.append(
                f"apoc.text.levenshteinSimilarity(toString(coalesce(n.{prop}, '')), ${param_name})"
            )

        if not similarity_calculations:
            return []

        # Build single query with all property comparisons
        query = f"""
        MATCH (n:{labels_clause})
        WITH n, ({' + '.join(similarity_calculations)}) / {len(similarity_calculations)} as similarity
        WHERE similarity > {self.thresholds['weak_match']}
        RETURN n, similarity
        ORDER BY similarity DESC
        LIMIT 5
        """

        try:
            with self.prod_db.session() as session:
                print(f"\nQuerying for similar nodes to {node.id}")
                print(f"Labels: {', '.join(node.labels)}")
                print("Properties:", staging_props)

                result = session.run(query, params)
                similar_nodes = []
                for record in result:
                    node_data = record["n"]
                    similar_nodes.append({
                        "id": node_data[self.UID_FIELD],
                        "labels": list(node_data.labels),
                        "properties": {k: v for k, v in dict(node_data).items()
                                    if k not in self.IGNORED_PROPERTIES},
                        "similarity": record["similarity"]
                    })

                print(f"Found {len(similar_nodes)} similar nodes for node {node}")
                for sim_node in similar_nodes:
                    print(f"- ID: {sim_node['id']}, Similarity: {sim_node['similarity']:.3f}")
                    print(sim_node)
                    if sim_node['similarity'] >= self.thresholds['auto_resolve']:
                        print("  (Auto-resolvable match)")

                return similar_nodes

        except Exception as e:
            print(f"Error querying similar nodes: {str(e)}")
            print(f"Query was: {query}")
            print(f"Parameters were: {params}")
            return []

    def run(self, state: ERState) -> ERState:
        """Run disambiguation checks on nodes"""
        print(f"Starting disambiguation for {len(state.staging_nodes)} nodes")

        new_processed_nodes = []
        new_count = 0
        resolved_count = 0
        review_count = 0

        for node in state.staging_nodes:
            try:
                similar_nodes = self.find_similar_nodes(node)

                if not similar_nodes:
                    print(f"New node detected: {node.id}")
                    print(f"Labels: {', '.join(node.labels)}")
                    print("Properties:", {k: v for k, v in node.properties.items()
                                       if k not in self.IGNORED_PROPERTIES})
                    result = NodeResolutionResult(
                        status=ResolutionStatus.NEW,
                        staging_node=node,
                        prod_node_id=None,
                        confidence=1.0,
                        issues=[]
                    )
                    new_count += 1

                elif similar_nodes[0]['similarity'] >= self.thresholds['auto_resolve']:
                    result = NodeResolutionResult(
                        status=ResolutionStatus.RESOLVED,
                        staging_node=node,
                        prod_node_id=similar_nodes[0]['id'],
                        confidence=similar_nodes[0]['similarity'],
                        issues=[]
                    )
                    resolved_count += 1

                else:
                    result = NodeResolutionResult(
                        status=ResolutionStatus.NEEDS_REVIEW,
                        staging_node=node,
                        prod_node_id=similar_nodes[0]['id'],
                        confidence=similar_nodes[0]['similarity'],
                        issues=[f"Multiple potential matches: {[n['id'] for n in similar_nodes]}"]
                        if len(similar_nodes) > 1
                        else [f"Low confidence match ({similar_nodes[0]['similarity']:.2f})"]
                    )
                    review_count += 1

                new_processed_nodes.append(result)

                if (result.status == ResolutionStatus.NEEDS_REVIEW or
                    result.confidence < self.thresholds['moderate_match']):
                    self.review_queue.enqueue(result)

            except Exception as e:
                print(f"Error processing node {node.id}: {str(e)}")
                state.errors.append(f"Error processing node {node.id}: {str(e)}")

        state.processed_nodes.extend(new_processed_nodes)

        print(f"\nDisambiguation complete:")
        print(f"- Total processed: {len(new_processed_nodes)}")
        print(f"- New nodes: {new_count}")
        print(f"- Resolved nodes: {resolved_count}")
        print(f"- Nodes needing review: {review_count}")

        return state