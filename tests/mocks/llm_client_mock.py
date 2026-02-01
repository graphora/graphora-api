"""Mock LLM client for London School TDD unit tests.

These mocks allow testing of components that depend on LLM extraction
without making actual API calls. They focus on verifying interactions
(methods called, parameters passed) and returning configured responses.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from dataclasses import dataclass, field


class MockLLMResponse:
    """Mock LLM extraction response.

    Provides dynamic attribute access for entity lists, matching
    the structure returned by real LLM extraction.

    Example:
        ```python
        response = MockLLMResponse(
            entities={"Company": [{"name": "Acme"}]},
            confidence=0.9
        )
        assert response.Company_list == [{"name": "Acme"}]
        assert response.confidence_score == 0.9
        ```
    """

    def __init__(
        self,
        entities: Dict[str, List[Any]] = None,
        confidence: float = 0.9,
    ):
        self._entities = entities or {}
        self.confidence_score = confidence

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "confidence_score":
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if name.endswith("_list"):
            entity_type = name[:-5]
            return self._entities.get(entity_type, [])

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"confidence_score": self.confidence_score}
        for entity_type, items in self._entities.items():
            result[f"{entity_type}_list"] = items
        return result


@dataclass
class MockEntityResolution:
    """Mock entity resolution result."""

    matching_ids: List[str]
    confidence_score: float = 0.9
    explanation: Optional[str] = None


class MockLLMClient:
    """Configurable mock LLM client for testing.

    Allows configuration of sequential responses for extraction methods
    and tracks all method calls for verification.

    Example:
        ```python
        client = MockLLMClient()
        client.configure_node_extraction(
            MockLLMResponse({"Company": [{"name": "Acme"}]}),
            MockLLMResponse({"Company": [{"name": "Beta"}]}),
        )

        # First call returns Acme
        result1 = await client.extract_nodes_from_chunk(...)
        # Second call returns Beta
        result2 = await client.extract_nodes_from_chunk(...)

        assert client.call_counts["extract_nodes_from_chunk"] == 2
        ```
    """

    def __init__(self):
        self._node_extractions: List[MockLLMResponse] = []
        self._relationship_extractions: List[MockLLMResponse] = []
        self._entity_resolutions: List[List[MockEntityResolution]] = []
        self._inferred_relationships: List[List[Any]] = []

        self._call_counts = {
            "extract_nodes_from_chunk": 0,
            "extract_relationships_from_chunk": 0,
            "extract_nodes_from_pdf": 0,
            "extract_relationships_from_pdf": 0,
            "resolve_entities": 0,
            "infer_relationship": 0,
        }

        self._call_args: Dict[str, List[Dict[str, Any]]] = {
            method: [] for method in self._call_counts.keys()
        }

        # Configuration for error simulation
        self._raise_on_method: Dict[str, Exception] = {}

    def configure_node_extraction(self, *responses: MockLLMResponse):
        """Configure sequential node extraction responses."""
        self._node_extractions = list(responses)

    def configure_relationship_extraction(self, *responses: MockLLMResponse):
        """Configure sequential relationship extraction responses."""
        self._relationship_extractions = list(responses)

    def configure_entity_resolution(self, *resolutions: List[MockEntityResolution]):
        """Configure entity resolution responses."""
        self._entity_resolutions = list(resolutions)

    def configure_inferred_relationships(self, *relationships: List[Any]):
        """Configure relationship inference responses."""
        self._inferred_relationships = list(relationships)

    def configure_error(self, method: str, exception: Exception):
        """Configure an error to be raised on method call."""
        self._raise_on_method[method] = exception

    def _check_error(self, method: str):
        """Check and raise configured error for method."""
        if method in self._raise_on_method:
            raise self._raise_on_method[method]

    def _get_response(
        self, responses: List[Any], call_count: int, default: Any = None
    ) -> Any:
        """Get response at current call index, wrapping to last if exhausted."""
        if not responses:
            return default
        idx = min(call_count - 1, len(responses) - 1)
        return responses[idx]

    async def extract_nodes_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> Any:
        """Mock node extraction from text chunk."""
        self._check_error("extract_nodes_from_chunk")
        self._call_counts["extract_nodes_from_chunk"] += 1
        self._call_args["extract_nodes_from_chunk"].append(
            {
                "chunk": chunk,
                "response_model": response_model,
                "ontology_yaml": ontology_yaml,
                "context": context,
                "user_id": user_id,
                "transform_id": transform_id,
                "document_usage_id": document_usage_id,
            }
        )

        return self._get_response(
            self._node_extractions,
            self._call_counts["extract_nodes_from_chunk"],
            MockLLMResponse({}),
        )

    async def extract_relationships_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> Any:
        """Mock relationship extraction from text chunk."""
        self._check_error("extract_relationships_from_chunk")
        self._call_counts["extract_relationships_from_chunk"] += 1
        self._call_args["extract_relationships_from_chunk"].append(
            {
                "chunk": chunk,
                "response_model": response_model,
                "ontology_yaml": ontology_yaml,
                "context": context,
                "user_id": user_id,
                "transform_id": transform_id,
                "document_usage_id": document_usage_id,
            }
        )

        return self._get_response(
            self._relationship_extractions,
            self._call_counts["extract_relationships_from_chunk"],
            MockLLMResponse({}),
        )

    async def extract_nodes_from_pdf(
        self,
        file_path: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> Any:
        """Mock node extraction from PDF file."""
        self._check_error("extract_nodes_from_pdf")
        self._call_counts["extract_nodes_from_pdf"] += 1
        self._call_args["extract_nodes_from_pdf"].append(
            {
                "file_path": file_path,
                "response_model": response_model,
                "ontology_yaml": ontology_yaml,
                "context": context,
                "user_id": user_id,
                "transform_id": transform_id,
                "document_usage_id": document_usage_id,
            }
        )

        return self._get_response(
            self._node_extractions,
            self._call_counts["extract_nodes_from_pdf"],
            MockLLMResponse({}),
        )

    async def extract_relationships_from_pdf(
        self,
        file_path: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> Any:
        """Mock relationship extraction from PDF file."""
        self._check_error("extract_relationships_from_pdf")
        self._call_counts["extract_relationships_from_pdf"] += 1
        self._call_args["extract_relationships_from_pdf"].append(
            {
                "file_path": file_path,
                "response_model": response_model,
                "ontology_yaml": ontology_yaml,
                "context": context,
                "user_id": user_id,
                "transform_id": transform_id,
                "document_usage_id": document_usage_id,
            }
        )

        return self._get_response(
            self._relationship_extractions,
            self._call_counts["extract_relationships_from_pdf"],
            MockLLMResponse({}),
        )

    async def resolve_entities(
        self,
        entity_type: str,
        node_dicts_str: str,
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> List[MockEntityResolution]:
        """Mock entity resolution call."""
        self._check_error("resolve_entities")
        self._call_counts["resolve_entities"] += 1
        self._call_args["resolve_entities"].append(
            {
                "entity_type": entity_type,
                "node_dicts_str": node_dicts_str,
                "user_id": user_id,
                "transform_id": transform_id,
                "document_usage_id": document_usage_id,
            }
        )

        return self._get_response(
            self._entity_resolutions,
            self._call_counts["resolve_entities"],
            [],
        )

    async def infer_relationship(
        self,
        source_node: Any,
        target_node: Any,
        ontology: Dict[str, Any],
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> List[Any]:
        """Mock relationship inference between nodes."""
        self._check_error("infer_relationship")
        self._call_counts["infer_relationship"] += 1
        self._call_args["infer_relationship"].append(
            {
                "source_node": source_node,
                "target_node": target_node,
                "ontology": ontology,
                "user_id": user_id,
                "transform_id": transform_id,
                "document_usage_id": document_usage_id,
            }
        )

        return self._get_response(
            self._inferred_relationships,
            self._call_counts["infer_relationship"],
            [],
        )

    @property
    def call_counts(self) -> Dict[str, int]:
        """Get method call counts."""
        return self._call_counts.copy()

    def get_call_args(self, method: str) -> List[Dict[str, Any]]:
        """Get all call arguments for a method."""
        return self._call_args.get(method, [])

    def assert_called(self, method: str, times: int = None):
        """Assert that a method was called.

        Args:
            method: Method name to check.
            times: Expected call count. If None, asserts at least once.
        """
        count = self._call_counts.get(method, 0)
        if times is not None:
            assert (
                count == times
            ), f"Expected {method} to be called {times} times, but was called {count} times"
        else:
            assert count > 0, f"Expected {method} to be called at least once"

    def assert_not_called(self, method: str):
        """Assert that a method was not called."""
        count = self._call_counts.get(method, 0)
        assert count == 0, f"Expected {method} to not be called, but was called {count} times"

    def reset(self):
        """Reset all call tracking and configured responses."""
        self._node_extractions = []
        self._relationship_extractions = []
        self._entity_resolutions = []
        self._inferred_relationships = []
        self._raise_on_method = {}

        for method in self._call_counts:
            self._call_counts[method] = 0
            self._call_args[method] = []
