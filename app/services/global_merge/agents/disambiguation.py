from app.schemas.global_merge import (
    DbNode, ERState, NodeResolutionResult, ResolutionStatus,
    ConflictType, NodeConflict, ConflictResolutionSuggestion
)
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

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

    def _analyze_property_differences(self, staging_props: Dict, prod_props: Dict) -> Tuple[float, List[str], Dict]:
        """Analyze property differences and calculate similarity score"""
        total_score = 0
        num_props = 0
        differences = []
        affected_props = {}

        for key in set(staging_props) | set(prod_props):
            if key in self.IGNORED_PROPERTIES:
                continue

            staging_val = str(staging_props.get(key, ''))
            prod_val = str(prod_props.get(key, ''))

            if staging_val != prod_val:
                differences.append(key)
                affected_props[key] = {
                    "staging": staging_val,
                    "prod": prod_val
                }

            if staging_val and prod_val:
                # Calculate similarity for non-empty values
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, staging_val, prod_val).ratio()
                total_score += similarity
                num_props += 1

        avg_score = total_score / num_props if num_props > 0 else 0
        return avg_score, differences, affected_props

    def _generate_suggestions(self, node: DbNode, similar_nodes: List[Dict], 
                            similarity_score: float, differences: List[str]) -> List[ConflictResolutionSuggestion]:
        """Generate suggestions for resolving conflicts"""
        suggestions = []

        if len(similar_nodes) > 1:
            # Multiple matches case
            best_match = similar_nodes[0]
            if best_match['similarity'] > self.thresholds['strong_match']:
                suggestions.append(ConflictResolutionSuggestion(
                    suggestion_type="merge",
                    description=f"Merge with node {best_match['id']} (highest similarity match)",
                    confidence=best_match['similarity'],
                    affected_properties=differences
                ))

        if similarity_score < self.thresholds['moderate_match']:
            # Low confidence case
            suggestions.append(ConflictResolutionSuggestion(
                suggestion_type="create_new",
                description="Create as a new node due to low similarity with existing nodes",
                confidence=1.0 - similarity_score,
                affected_properties=[]
            ))

        if differences:
            # Property conflicts case
            suggestions.append(ConflictResolutionSuggestion(
                suggestion_type="selective_merge",
                description="Selectively merge properties, keeping the most detailed/recent values",
                confidence=0.8,
                affected_properties=differences
            ))

        return suggestions

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
                logger.info(f"Querying for similar nodes to {node.id}")
                logger.info(f"Labels: {', '.join(node.labels)}")
                logger.info("Properties:", staging_props)

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

                logger.info(f"Found {len(similar_nodes)} similar nodes")
                return similar_nodes

        except Exception as e:
            logger.error(f"Error querying similar nodes: {str(e)}")
            logger.error(f"Query was: {query}")
            logger.error(f"Parameters were: {params}")
            return []

    def run(self, state: ERState) -> ERState:
        """Run disambiguation checks on nodes"""
        logger.info(f"Starting disambiguation for {len(state.staging_nodes)} nodes")

        for node in state.staging_nodes:
            try:
                similar_nodes = self.find_similar_nodes(node)

                if not similar_nodes:
                    # New node case
                    result = NodeResolutionResult(
                        status=ResolutionStatus.NEW,
                        staging_node=node,
                        prod_node_id=None,
                        confidence=1.0,
                        issues=[],
                        conflicts=[],
                        suggested_resolution=None
                    )
                    state.new_nodes.append(node)

                else:
                    # Analyze the best match
                    best_match = similar_nodes[0]
                    similarity_score = best_match['similarity']
                    
                    # Analyze property differences
                    score, differences, affected_props = self._analyze_property_differences(
                        node.properties, best_match['properties']
                    )

                    if similarity_score >= self.thresholds['auto_resolve'] and not differences:
                        # Clear match case
                        result = NodeResolutionResult(
                            status=ResolutionStatus.RESOLVED,
                            staging_node=node,
                            prod_node_id=best_match['id'],
                            confidence=similarity_score,
                            issues=[],
                            conflicts=[],
                            suggested_resolution=None
                        )
                        state.updated_nodes.append({
                            "staging": node,
                            "prod": best_match
                        })

                    else:
                        # Create conflict with suggestions
                        suggestions = self._generate_suggestions(
                            node, similar_nodes, similarity_score, differences
                        )
                        
                        conflict = NodeConflict(
                            conflict_type=(
                                ConflictType.MULTIPLE_MATCHES if len(similar_nodes) > 1
                                else ConflictType.LOW_CONFIDENCE if similarity_score < self.thresholds['moderate_match']
                                else ConflictType.PROPERTY_CONFLICT
                            ),
                            staging_node_id=node.id,
                            prod_node_ids=[n['id'] for n in similar_nodes],
                            description=self._get_conflict_description(
                                similar_nodes, similarity_score, differences
                            ),
                            suggestions=suggestions,
                            properties_affected=affected_props
                        )

                        result = NodeResolutionResult(
                            status=ResolutionStatus.NEEDS_REVIEW,
                            staging_node=node,
                            prod_node_id=best_match['id'],
                            confidence=similarity_score,
                            conflicts=[conflict],
                            suggested_resolution=suggestions[0] if suggestions else None
                        )

                        state.add_conflict(conflict)

                state.processed_nodes.append(result)

            except Exception as e:
                error_msg = f"Error processing node {node.id}: {str(e)}"
                logger.error(error_msg)
                state.errors.append(error_msg)

        logger.info(f"Disambiguation complete: {len(state.processed_nodes)} nodes processed")
        return state

    def _get_conflict_description(self, similar_nodes: List[Dict], 
                                similarity_score: float, 
                                differences: List[str]) -> str:
        """Generate a human-readable conflict description"""
        if len(similar_nodes) > 1:
            return (f"Found {len(similar_nodes)} potential matches. "
                   f"Best match has {similarity_score:.2f} similarity score.")
        elif similarity_score < self.thresholds['moderate_match']:
            return (f"Low confidence match (score: {similarity_score:.2f}). "
                   "The existing node might be different from this one.")
        else:
            return (f"Found matching node but there are differences in: "
                   f"{', '.join(differences)}")