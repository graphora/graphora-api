from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Tuple
from datetime import datetime
import json
import unicodedata
from typing import Dict, Any
import logging
from app.schemas.local import LocalNode, LocalEdge

class BERTResolver:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.similarity_threshold = 0.93
        self.type_groups = {}
        self.relationship_groups = {}
        self.id_map = {}  # Global id_map for tracking all node merges

        # Setup logging
        self.logger = logging.getLogger('BERTResolver')
        self.logger.setLevel(logging.INFO)

        # Create file handlers for different log types
        self.setup_loggers()

    def setup_loggers(self):
        # Merged nodes logger
        merged_handler = logging.FileHandler('merged_nodes.json', mode='w')
        self.merged_logger = logging.getLogger('merged_nodes')
        self.merged_logger.setLevel(logging.INFO)
        self.merged_logger.addHandler(merged_handler)

        # Unmerged nodes logger
        unmerged_handler = logging.FileHandler('unmerged_nodes.json', mode='w')
        self.unmerged_logger = logging.getLogger('unmerged_nodes')
        self.unmerged_logger.setLevel(logging.INFO)
        self.unmerged_logger.addHandler(unmerged_handler)

        # Resolution metrics logger
        metrics_handler = logging.FileHandler('resolution_metrics.json', mode='w')
        self.metrics_logger = logging.getLogger('resolution_metrics')
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.addHandler(metrics_handler)

    def log_node_comparison(self, node1: LocalNode, node2: LocalNode,
                       similarity: float, threshold: float, merged: bool):
        """Log detailed comparison information with threshold"""
        comparison_data = {
            'node1': {
                'id': node1.id,
                'type': node1.type_,
                'properties': node1.properties
            },
            'node2': {
                'id': node2.id,
                'type': node2.type_,
                'properties': node2.properties
            },
            'similarity_score': similarity,
            'threshold': threshold,
            'merged': merged,
            'timestamp': datetime.now().isoformat()
        }

        if merged:
            self.merged_logger.info(json.dumps(comparison_data))
        else:
            self.unmerged_logger.info(json.dumps(comparison_data))

    def log_resolution_metrics(self, type_: str, initial_count: int, final_count: int, merges: List[Dict]):
        """Log resolution metrics for a type group"""
        metrics = {
            'node_type': type_,
            'initial_count': initial_count,
            'final_count': final_count,
            'merge_count': initial_count - final_count,
            'merge_percentage': ((initial_count - final_count) / initial_count) * 100 if initial_count > 0 else 0,
            'merges': merges,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics_logger.info(json.dumps(metrics))

    def get_embeddings(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1)

    def group_by_type(self, nodes: List[LocalNode]):
        """Group nodes by their type, maintaining original order"""
        self.type_groups.clear()
        for node in nodes:
            if node.type_ not in self.type_groups:
                self.type_groups[node.type_] = []
            self.type_groups[node.type_].append(node)

    def group_by_relationship(self, edges: List[LocalEdge]):
        self.relationship_groups.clear()
        for edge in edges:
            if edge.relationship not in self.relationship_groups:
                self.relationship_groups[edge.relationship] = []
            self.relationship_groups[edge.relationship].append(edge)

    def normalize_text(self, text: str) -> str:
        """
        Comprehensive text normalization that:
        1. Handles unicode variations
        2. Preserves meaningful characters
        3. Normalizes whitespace
        """
        if not isinstance(text, str):
            text = str(text)

        # Convert to NFKC normalized form (handles complex unicode)
        text = unicodedata.normalize('NFKC', text)

        # Convert to lowercase for comparison
        text = text.lower()

        # Replace various types of quotes and apostrophes with standard ones
        quotes_map = {
            '"': '"',  # double quotes
            '"': '"',
            '"': '"',
            ''': "'",  # single quotes
            ''': "'",
            '`': "'",
            '´': "'",
        }
        for char, replacement in quotes_map.items():
            text = text.replace(char, replacement)

        # Handle special characters while preserving meaningful ones
        # Keep alphanumeric, preserve certain punctuation
        chars = []
        for char in text:
            if char.isalnum():
                chars.append(char)
            elif char in ".-/":  # Preserve meaningful separators
                chars.append(" ")
            elif char.isspace():
                chars.append(" ")
            # Ignore other special characters

        # Normalize whitespace
        text = ''.join(chars)
        text = ' '.join(text.split())

        return text.strip()

    def compute_node_signature(self, node: LocalNode) -> torch.Tensor:
        values = []
        for v in node.properties.values():
            if v and not isinstance(v, bool):  # Exclude empty values and booleans
                norm_value = self.normalize_text(str(v))
                if norm_value:
                    values.append(norm_value)

        signature_parts = [
            f"Type: {node.type_}",  # Type is important
            "Values: " + " | ".join(values)  # Focus on actual values
        ]

        full_text = " ".join(signature_parts)
        return self.get_embeddings(full_text)

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        return torch.cosine_similarity(emb1, emb2, dim=1).item()

    def merge_properties(self, props1: Dict, props2: Dict) -> Dict:
        """Merge properties with proper handling of merged_ids"""
        merged = props1.copy()

        # Handle special _merged_ids property
        merged_ids = set(props1.get('_merged_ids', []))
        merged_ids.update(props2.get('_merged_ids', []))
        if merged_ids:
            merged['_merged_ids'] = list(merged_ids)

        # Merge other properties
        for k, v2 in props2.items():
            if k != '_merged_ids':  # Skip _merged_ids as it's handled above
                if k not in merged or not merged[k]:
                    merged[k] = v2
                elif v2 and len(str(v2)) > len(str(merged[k])):
                    merged[k] = v2

        return merged


    def update_edge_references(self, edge_id: str, removed_id: str) -> str:
        """Update edge endpoint references based on node merges"""
        return self.id_map.get(edge_id, edge_id)

    def get_dynamic_threshold(self, node1_props: Dict[str, Any], node2_props: Dict[str, Any]) -> float:
        """
        Get dynamic threshold based on:
        1. Property overlap
        2. Text length/complexity
        3. Value patterns
        """
        base_threshold = self.similarity_threshold

        # Get normalized property values
        values1 = {k: self.normalize_text(str(v))
                  for k, v in node1_props.items()
                  if v is not None and k != '_uid_'}
        values2 = {k: self.normalize_text(str(v))
                  for k, v in node2_props.items()
                  if v is not None and k != '_uid_'}

        # Calculate property overlap ratio
        common_keys = set(values1.keys()) & set(values2.keys())
        total_keys = set(values1.keys()) | set(values2.keys())
        overlap_ratio = len(common_keys) / len(total_keys) if total_keys else 0

        # Adjust threshold based on overlap
        if overlap_ratio > 0.7:  # High property overlap
            threshold_adjustment = -0.05
        elif overlap_ratio < 0.3:  # Low property overlap
            threshold_adjustment = +0.05
        else:
            threshold_adjustment = 0

        # Consider text length/complexity
        avg_length1 = sum(len(v) for v in values1.values()) / len(values1) if values1 else 0
        avg_length2 = sum(len(v) for v in values2.values()) / len(values2) if values2 else 0

        # More lenient for very short texts
        if avg_length1 < 10 and avg_length2 < 10:
            threshold_adjustment -= 0.03

        return max(0.75, min(0.98, base_threshold + threshold_adjustment))

    def resolve_nodes(self, nodes: List[LocalNode]) -> List[LocalNode]:
        """Resolve nodes with merged node tracking"""
        self.logger.info(f"Starting node resolution with {len(nodes)} nodes")

        self.group_by_type(nodes)
        resolved_nodes = []
        processed_ids = set()
        self.id_map.clear()

        for type_, type_nodes in self.type_groups.items():
            initial_count = len(type_nodes)

            # Group nodes that should be merged
            node_groups = []  # List to hold groups of nodes that should be merged
            current_group = []

            node_embeddings = []
            for node in type_nodes:
                if node.id not in processed_ids:
                    emb = self.compute_node_signature(node)
                    node_embeddings.append((node, emb))

            i = 0
            while i < len(node_embeddings):
                node1, emb1 = node_embeddings[i]

                if node1.id in processed_ids:
                    i += 1
                    continue

                # Start new group with this node
                current_group = [node1]
                merged_props = node1.properties.copy()
                merged_ids = [node1.id]

                j = i + 1
                while j < len(node_embeddings):
                    node2, emb2 = node_embeddings[j]

                    if node2.id not in processed_ids:
                        similarity = self.compute_similarity(emb1, emb2)
                        threshold = self.get_dynamic_threshold(node1.properties, node2.properties)

                        # self.log_node_comparison(
                        #     node1, node2, similarity,
                        #     threshold,
                        #     similarity >= threshold
                        # )

                        if similarity >= threshold:
                            self.logger.info(
                                f"Merging nodes: {node2.id} into {node1.id} "
                                f"(similarity: {similarity:.4f}, threshold: {threshold})"
                            )
                            current_group.append(node2)
                            merged_props = self.merge_properties(merged_props, node2.properties)
                            merged_ids.append(node2.id)
                            processed_ids.add(node2.id)

                    j += 1

                # Store the merged ids regardless of merging
                if merged_ids:
                    merged_props['_merged_ids'] = merged_ids

                # Create merged node
                merged_node = node1.__class__(
                    id=node1.id,
                    type_=node1.type_,
                    properties=merged_props,
                    metadata=node1.metadata
                )

                resolved_nodes.append(merged_node)
                processed_ids.add(node1.id)

                # Map all merged IDs to the primary node
                for merged_id in merged_ids:
                    self.id_map[merged_id] = node1.id

                i += 1

        # Debug information
        print("\nResolved nodes:")
        for node in resolved_nodes:
            print(f"Node: {node.id}")
            print(f"Merged IDs: {node.properties.get('_merged_ids', [])}")
            print(f"Properties: {node.properties}")
            print("---")

        return resolved_nodes


    def resolve_edges(self, edges: List[LocalEdge]) -> List[LocalEdge]:
        self.group_by_relationship(edges)
        resolved_edges = []
        edge_signatures = set()

        for rel_type, rel_edges in self.relationship_groups.items():
            for edge in rel_edges:
                # Update both endpoints using the id_map
                from_id = self.update_edge_references(edge.from_, edge.from_)
                to_id = self.update_edge_references(edge.to, edge.to)

                # Create unique signature for deduplication
                edge_sig = f"{from_id}|{rel_type}|{to_id}"

                if edge_sig not in edge_signatures:
                    new_edge = LocalEdge(
                        from_=from_id,
                        to=to_id,
                        relationship=rel_type,
                        properties=edge.properties.copy(),
                        metadata=edge.metadata.copy()
                    )
                    resolved_edges.append(new_edge)
                    edge_signatures.add(edge_sig)
                else:
                    # If edge exists, merge properties with existing edge
                    existing_edge = next(e for e in resolved_edges
                                      if f"{e.from_}|{e.relationship}|{e.to}" == edge_sig)
                    existing_edge.properties = self.merge_properties(
                        existing_edge.properties,
                        edge.properties
                    )

        return resolved_edges

def resolve_with_bert(nodes: List[LocalNode], edges: List[LocalEdge]) -> Tuple[List[LocalNode], List[LocalEdge]]:
    resolver = BERTResolver()
    resolved_nodes = resolver.resolve_nodes(nodes)
    resolved_edges = resolver.resolve_edges(edges)
    return resolved_nodes, resolved_edges