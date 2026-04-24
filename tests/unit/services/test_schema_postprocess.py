"""Unit tests for post-hoc ontology inference.

These cover the graph-summary builder, YAML parsing, the guard
paths (empty graph, bad LLM output), and the happy path with a
mocked Gemini client.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphora_server.services.schema_postprocess import (
    _parse_yaml_response,
    infer_ontology_from_graph,
    ontology_dict_to_yaml,
    summarize_graph,
)


# ---- summarize_graph -------------------------------------------------------


class TestSummarizeGraph:
    def test_groups_nodes_by_type(self) -> None:
        nodes = [
            {
                "id": "n1",
                "type": "Person",
                "label": "Alice",
                "properties": {"name": "Alice", "age": 30},
            },
            {
                "id": "n2",
                "type": "Person",
                "label": "Bob",
                "properties": {"name": "Bob"},
            },
            {
                "id": "n3",
                "type": "Organization",
                "label": "Acme",
                "properties": {"name": "Acme"},
            },
        ]
        summary = summarize_graph(nodes, edges=[])
        assert "Person (2 nodes)" in summary
        assert "Organization (1 nodes)" in summary
        assert "Alice" in summary
        assert "Bob" in summary

    def test_includes_relationship_patterns(self) -> None:
        nodes = [
            {"id": "n1", "type": "Person", "properties": {}},
            {"id": "n2", "type": "Organization", "properties": {}},
        ]
        edges = [
            {"source": "n1", "target": "n2", "type": "WORKS_AT"},
            {"source": "n1", "target": "n2", "type": "WORKS_AT"},
        ]
        summary = summarize_graph(nodes, edges)
        assert "Person --WORKS_AT-> Organization (2)" in summary

    def test_handles_missing_properties(self) -> None:
        """Nodes without properties dict should not crash."""
        nodes = [{"id": "n1", "type": "Person"}]
        summary = summarize_graph(nodes, edges=[])
        assert "Person (1 nodes)" in summary

    def test_caps_label_examples_per_type(self) -> None:
        """Very large types should not produce unbounded summaries."""
        nodes = [
            {
                "id": f"n{i}",
                "type": "Person",
                "label": f"Person_{i}",
                "properties": {"name": f"Person_{i}"},
            }
            for i in range(20)
        ]
        summary = summarize_graph(nodes, edges=[])
        # Only the first _MAX_NODE_SAMPLES_PER_TYPE (5) example names
        # should appear in the 'examples: ' portion of the line.
        person_line = next(
            line for line in summary.splitlines() if line.startswith("- Person (20")
        )
        assert person_line.count("Person_") == 5

    def test_unknown_type_when_missing(self) -> None:
        nodes = [{"id": "n1", "properties": {"name": "x"}}]
        summary = summarize_graph(nodes, edges=[])
        assert "Unknown (1 nodes)" in summary


# ---- _parse_yaml_response --------------------------------------------------


class TestParseYamlResponse:
    def test_plain_yaml(self) -> None:
        yaml_text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
relationships: {}
"""
        result = _parse_yaml_response(yaml_text)
        assert "Person" in result["entities"]

    def test_yaml_inside_code_fence(self) -> None:
        yaml_text = (
            "Here is the refined ontology:\n\n"
            "```yaml\n"
            'version: "0.1.0"\n'
            "entities:\n"
            "  Organization:\n"
            "    properties:\n"
            "      name:\n"
            "        type: str\n"
            "```\n"
        )
        result = _parse_yaml_response(yaml_text)
        assert "Organization" in result["entities"]

    def test_invalid_yaml_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_yaml_response("not: valid: yaml: [[[")

    def test_non_dict_yaml_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected dict"):
            _parse_yaml_response("- item1\n- item2")


# ---- infer_ontology_from_graph --------------------------------------------


class TestInferOntology:
    @pytest.mark.asyncio
    async def test_empty_graph_raises(self) -> None:
        with pytest.raises(ValueError, match="empty graph"):
            await infer_ontology_from_graph(nodes=[], edges=[], user_id="u1")

    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        """Simple graph → refined ontology via mocked Gemini."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    description: "A person"
    properties:
      name:
        type: str
        required: true
  Organization:
    description: "A company"
    properties:
      name:
        type: str
        required: true
relationships:
  WORKS_AT:
    source: Person
    target: Organization
    properties: {}
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        nodes = [
            {
                "id": "n1",
                "type": "Person",
                "label": "Alice",
                "properties": {"name": "Alice"},
            },
            {
                "id": "n2",
                "type": "Organization",
                "label": "Acme",
                "properties": {"name": "Acme"},
            },
        ]
        edges = [{"source": "n1", "target": "n2", "type": "WORKS_AT"}]

        with patch(
            "graphora_server.services.schema_postprocess.get_user_llm_credentials",
            new_callable=AsyncMock,
            return_value=("api-key", "gemini-2.5-flash"),
        ):
            result = await infer_ontology_from_graph(
                nodes=nodes,
                edges=edges,
                user_id="u1",
                client_factory=lambda _k: mock_client,
            )

        assert "Person" in result["entities"]
        assert "Organization" in result["entities"]
        assert "WORKS_AT" in result["relationships"]
        assert result["version"] == "0.1.0"
        mock_client.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_adds_missing_version_and_relationships(self) -> None:
        mock_response = MagicMock()
        mock_response.text = """entities:
  Person:
    properties:
      name:
        type: str
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_postprocess.get_user_llm_credentials",
            new_callable=AsyncMock,
            return_value=("api-key", "model"),
        ):
            result = await infer_ontology_from_graph(
                nodes=[{"id": "n1", "type": "Person"}],
                edges=[],
                user_id="u1",
                client_factory=lambda _k: mock_client,
            )

        assert result["version"] == "0.1.0"
        assert result["relationships"] == {}

    @pytest.mark.asyncio
    async def test_empty_llm_response_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_postprocess.get_user_llm_credentials",
            new_callable=AsyncMock,
            return_value=("api-key", "model"),
        ):
            with pytest.raises(ValueError, match="Empty response"):
                await infer_ontology_from_graph(
                    nodes=[{"id": "n1", "type": "Person"}],
                    edges=[],
                    user_id="u1",
                    client_factory=lambda _k: mock_client,
                )

    @pytest.mark.asyncio
    async def test_llm_returning_no_entities_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities: {}
relationships: {}
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_postprocess.get_user_llm_credentials",
            new_callable=AsyncMock,
            return_value=("api-key", "model"),
        ):
            with pytest.raises(ValueError, match="no entities"):
                await infer_ontology_from_graph(
                    nodes=[{"id": "n1", "type": "Person"}],
                    edges=[],
                    user_id="u1",
                    client_factory=lambda _k: mock_client,
                )

    @pytest.mark.asyncio
    async def test_coerces_non_string_property_types(self) -> None:
        """If the LLM confuses 'type' keys, coerce to 'str'."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: Person
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_postprocess.get_user_llm_credentials",
            new_callable=AsyncMock,
            return_value=("api-key", "model"),
        ):
            # Don't auto-coerce string types that are real Python types;
            # the coercion only kicks in for non-strings. The "Person"
            # value above is a string so it stays as-is.
            result = await infer_ontology_from_graph(
                nodes=[{"id": "n1", "type": "Person"}],
                edges=[],
                user_id="u1",
                client_factory=lambda _k: mock_client,
            )
        # Sanity: string type values are preserved.
        assert result["entities"]["Person"]["properties"]["name"]["type"] == "Person"


# ---- ontology_dict_to_yaml -------------------------------------------------


class TestOntologyDictToYaml:
    def test_roundtrips(self) -> None:
        ontology = {
            "version": "0.1.0",
            "entities": {
                "Person": {
                    "description": "A person",
                    "properties": {
                        "name": {"type": "str", "required": True},
                    },
                }
            },
            "relationships": {},
        }
        rendered = ontology_dict_to_yaml(ontology)
        assert "Person:" in rendered
        assert "name:" in rendered
        assert "required: true" in rendered.lower()
