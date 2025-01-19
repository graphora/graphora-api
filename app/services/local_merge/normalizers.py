from typing import Dict, List, Any, Union
import re
from datetime import datetime
from app.schemas.local import LocalNode, LocalEdge

class PropertyNormalizer:
    def __init__(self):
        self.unknown_values = {'<UNKNOWN>', 'UNKNOWN', 'N/A', 'None', ''}

    def normalize_property_value(self, value: Any) -> Any:
        if isinstance(value, str):
            # Handle unknown values
            if value.strip() in self.unknown_values:
                return None

            # Try date conversion
            # date_value = self.try_parse_date(value)
            # if date_value:
            #     return date_value

            # Try numeric conversion
            numeric_value = self.try_parse_numeric(value)
            if numeric_value is not None:
                return numeric_value

            # Clean text
            return self.clean_text(value)

        return value

    def try_parse_date(self, value: str) -> Union[datetime, None]:
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{4}'  # Year only
        ]

        for pattern in date_patterns:
          match = re.search(pattern, value)
          if match:
            try:
              date_str = match.group()
              if len(date_str) == 4:  # Year only
                  return datetime.strptime(date_str, '%Y')
              elif '-' in date_str:
                  return datetime.strptime(date_str, '%Y-%m-%d')
              else:
                  return datetime.strptime(date_str, '%m/%d/%Y')
            except ValueError:
              continue
        return None

    def try_parse_numeric(self, value: str) -> Union[float, int, None]:
        # Remove common non-numeric characters
        clean_value = re.sub(r'[,$%]', '', value)

        try:
            if '.' in clean_value:
                return float(clean_value)
            numeric_value = int(clean_value)
            return numeric_value
        except ValueError:
            return None

    def clean_text(self, text: str) -> str:
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;!?-]', '', text)
        return text.strip()

    def normalize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}

        for key, value in properties.items():
            # Normalize key names
            clean_key = re.sub(r'([A-Z])', r'_\1', key).lower().strip('_')

            # Normalize values
            normalized_value = self.normalize_property_value(value)

            if normalized_value is not None:
                normalized[clean_key] = normalized_value

        return normalized

def normalize_graph_data(nodes: List[LocalNode], edges: List[LocalEdge]):
    normalizer = PropertyNormalizer()

    # Normalize node properties
    for node in nodes:
        node.properties = normalizer.normalize_properties(node.properties)
        if node.metadata:
            node.metadata = normalizer.normalize_properties(node.metadata)

    # Normalize edge properties
    for edge in edges:
        edge.properties = normalizer.normalize_properties(edge.properties)
        if edge.metadata:
            edge.metadata = normalizer.normalize_properties(edge.metadata)

    return nodes, edges