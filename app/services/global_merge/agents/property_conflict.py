from app.schemas.global_merge import DbNode, ResolutionStatus, ERState
from typing import List, Dict

class PropertyConflictAgent:
    """Agent for detecting and resolving property conflicts"""

    def __init__(self, prod_db, review_queue):
        self.prod_db = prod_db
        self.UID_FIELD = '_uid_'
        self.review_queue = review_queue
        self.IGNORED_PROPERTIES = {'_merged_ids', self.UID_FIELD}

    def _filter_properties(self, properties: Dict) -> Dict:
        """Filter out ignored properties"""
        return {k: v for k, v in properties.items()
                if k not in self.IGNORED_PROPERTIES}

    def check_property_conflicts(self, staging_node: DbNode, prod_node_id: str) -> List[str]:
        """Check for property conflicts between staging and production nodes"""
        query = f"""
        MATCH (n)
        WHERE n.{self.UID_FIELD} = $prod_id
        RETURN n
        """

        conflicts = []
        try:
            with self.prod_db.session() as session:
                result = session.run(query, prod_id=prod_node_id)
                prod_node = result.single()
                if prod_node:
                    # Filter properties for both nodes
                    prod_data = self._filter_properties(dict(prod_node["n"]))
                    staging_data = self._filter_properties(staging_node.properties)

                    # Compare filtered properties
                    for key, value in staging_data.items():
                        if key in prod_data and prod_data[key] != value:
                            conflicts.append(
                                f"Property conflict: {key} differs - "
                                f"staging: {value}, prod: {prod_data[key]}"
                            )

                    # Check for label differences
                    prod_labels = set(prod_node["n"].labels)
                    staging_labels = set(staging_node.labels)
                    if prod_labels != staging_labels:
                        conflicts.append(
                            f"Label mismatch: staging {staging_labels} vs prod {prod_labels}"
                        )

                return conflicts
        except Exception as e:
            print(f"Error checking conflicts: {str(e)}")
            return [f"Error checking conflicts: {str(e)}"]

    def run(self, state: ERState) -> ERState:
        """Check for property conflicts in matched nodes"""
        print(f"Starting property conflict resolution for {len(state.processed_nodes)} nodes")

        processed_count = 0
        conflict_count = 0
        review_count = 0

        for result in state.processed_nodes:
            try:
                if result.prod_node_id:  # Only check nodes with matches
                    conflicts = self.check_property_conflicts(
                        result.staging_node,
                        result.prod_node_id
                    )

                    if conflicts:
                        conflict_count += 1
                        result.issues.extend(conflicts)
                        result.confidence *= 0.8  # Reduce confidence

                        if result.confidence < 0.85:  # Configurable threshold
                            result.status = ResolutionStatus.NEEDS_REVIEW
                            self.review_queue.enqueue(result)
                            review_count += 1

                    processed_count += 1

            except Exception as e:
                error_msg = f"Error checking conflicts for node {result.staging_node.id}: {str(e)}"
                print(error_msg)
                state.errors.append(error_msg)

        print("\nProperty conflict resolution complete:")
        print(f"- Nodes processed: {processed_count}")
        print(f"- Conflicts found: {conflict_count}")
        print(f"- Sent for review: {review_count}")

        return state